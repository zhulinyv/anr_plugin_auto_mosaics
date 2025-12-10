import gc
import os
from collections import namedtuple
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps
from segment_anything import SamPredictor, sam_model_registry
from ultralytics import YOLO

from utils.logger import logger

SEG = namedtuple(
    "SEG",
    ["cropped_image", "cropped_mask", "confidence", "crop_region", "bbox", "label", "control_net_wrapper"],
    defaults=[None],
)


def tensor2pil(image: torch.Tensor) -> Image.Image:
    if image.ndim != 4:
        raise ValueError(f"Expected NHWC tensor, found {image.ndim} dimensions")
    return Image.fromarray(np.clip(255.0 * image.cpu().numpy().squeeze(0), 0, 255).astype(np.uint8))


def load_image_to_tensor(image_path: str) -> torch.Tensor:
    img = Image.open(image_path)
    img = ImageOps.exif_transpose(img)
    if img.mode == "I":
        img = img.point(lambda i: i * (1 / 255))
    img = img.convert("RGB")
    img_np = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(img_np)[None, :]


def mask_to_tensor_visual(mask: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
    if isinstance(mask, np.ndarray):
        mask = torch.from_numpy(mask)
    mask = mask.float()
    if len(mask.shape) == 4:
        mask = mask.squeeze(0).squeeze(0)
    elif len(mask.shape) == 3:
        mask = mask.squeeze(0)
    if mask.ndim == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.ndim == 3:
        mask = mask.unsqueeze(0)
    result = mask.movedim(1, -1)
    result = result.expand(-1, -1, -1, 3)
    return result


def save_mask_as_image(mask_tensor_tuple: Tuple[torch.Tensor], filename: str = "result_mask.png"):
    tensor_result = mask_tensor_tuple[0]
    if tensor_result.is_cuda:
        tensor_result = tensor_result.cpu()
    if tensor_result.requires_grad:
        tensor_result = tensor_result.detach()
    numpy_array = tensor_result[0].numpy()
    if numpy_array.dtype in [np.float32, np.float64]:
        numpy_array = (np.clip(numpy_array, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(numpy_array).save(filename)


def normalize_region(limit, startp, size):
    if startp < 0:
        new_endp = min(limit, size)
        new_startp = 0
    elif startp + size > limit:
        new_startp = max(0, limit - size)
        new_endp = limit
    else:
        new_startp = startp
        new_endp = min(limit, startp + size)
    return int(new_startp), int(new_endp)


def make_crop_region(w, h, bbox, crop_factor, crop_min_size=None):
    x1, y1, x2, y2 = bbox
    bbox_w, bbox_h = x2 - x1, y2 - y1
    crop_w = bbox_w * crop_factor
    crop_h = bbox_h * crop_factor
    if crop_min_size is not None:
        crop_w = max(crop_min_size, crop_w)
        crop_h = max(crop_min_size, crop_h)
    center_x = x1 + bbox_w / 2
    center_y = y1 + bbox_h / 2
    new_x1 = int(center_x - crop_w / 2)
    new_y1 = int(center_y - crop_h / 2)
    new_x1, new_x2 = normalize_region(w, new_x1, crop_w)
    new_y1, new_y2 = normalize_region(h, new_y1, crop_h)
    return [new_x1, new_y1, new_x2, new_y2]


def crop_tensor(tensor: Union[np.ndarray, torch.Tensor], region):
    x1, y1, x2, y2 = region
    if isinstance(tensor, np.ndarray):
        if tensor.ndim in (3, 2):
            return tensor[y1:y2, x1:x2]
    elif isinstance(tensor, torch.Tensor):
        if tensor.ndim == 4:
            return tensor[:, y1:y2, x1:x2, :]
    return tensor


def center_of_bbox(bbox):
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    return bbox[0] + w / 2, bbox[1] + h / 2


def combine_masks(masks: List[np.ndarray]) -> Optional[torch.Tensor]:
    if not masks:
        return None
    initial_mask = masks[0].astype(np.uint8)
    combined_mask = initial_mask
    for i in range(1, len(masks)):
        curr_mask = masks[i].astype(np.uint8)
        if combined_mask.shape == curr_mask.shape:
            combined_mask = cv2.bitwise_or(combined_mask, curr_mask)
    return torch.from_numpy(combined_mask)


def dilate_mask_array(mask: np.ndarray, dilation_factor: int, iter: int = 1) -> np.ndarray:
    if dilation_factor == 0:
        return mask
    kernel = np.ones((abs(dilation_factor), abs(dilation_factor)), np.uint8)
    mask = mask.astype(np.uint8)
    if dilation_factor > 0:
        return cv2.dilate(mask, kernel, iterations=iter)
    else:
        return cv2.erode(mask, kernel, iterations=iter)


class UltraBBoxDetector:
    def __init__(self, model_path: str):
        self.bbox_model = YOLO(model_path, verbose=False)

        self.yolo_device = "cpu"
        if torch.cuda.is_available():
            self.yolo_device = "cuda"
            self.bbox_model.to(self.yolo_device)

    def inference_bbox(self, image: Image.Image, confidence: float = 0.3, device: str = ""):
        pred = self.bbox_model(image, conf=confidence, device=self.yolo_device, verbose=False)
        bboxes = pred[0].boxes.xyxy.cpu().numpy()

        cv2_image = np.array(image)
        h, w = cv2_image.shape[:2]

        segms = []
        for x0, y0, x1, y1 in bboxes:
            mask = np.zeros((h, w), np.uint8)
            cv2.rectangle(mask, (int(x0), int(y0)), (int(x1), int(y1)), 255, -1)
            segms.append(mask.astype(bool))

        n = bboxes.shape[0]
        if n == 0:
            return [[], [], [], []]

        results = [[], [], [], []]
        for i in range(n):
            cls_id = int(pred[0].boxes[i].cls.item())
            results[0].append(pred[0].names[cls_id])
            results[1].append(bboxes[i])
            results[2].append(segms[i])
            results[3].append(pred[0].boxes[i].conf.cpu().numpy())
        return results

    def detect(
        self, image_tensor: torch.Tensor, threshold=0.5, dilation=10, crop_factor=3.0, drop_size=1, detailer_hook=None
    ):
        pil_img = tensor2pil(image_tensor)
        detected_results = self.inference_bbox(pil_img, threshold)

        labels_list, bboxes_list, masks_list, conf_list = detected_results

        processed_masks = []
        for mask, bbox, conf in zip(masks_list, bboxes_list, conf_list):
            m_float = mask.astype(np.float32)
            if dilation != 0:
                m_float = dilate_mask_array(m_float, dilation)
            processed_masks.append((bbox, m_float, conf))

        seg_items = []
        h, w = pil_img.height, pil_img.width
        img_np = np.array(pil_img)

        for i, (bbox, mask, conf) in enumerate(processed_masks):
            label = labels_list[i]
            x1, y1, x2, y2 = bbox

            if (x2 - x1) <= drop_size or (y2 - y1) <= drop_size:
                continue

            crop_region = make_crop_region(w, h, bbox, crop_factor)

            if detailer_hook is not None:
                crop_region = detailer_hook.post_crop_region(w, h, bbox, crop_region)

            cropped_image_np = crop_tensor(img_np, crop_region)
            cropped_image_tensor = torch.from_numpy(cropped_image_np).float() / 255.0
            if cropped_image_tensor.ndim == 3:
                cropped_image_tensor = cropped_image_tensor.unsqueeze(0)

            cropped_mask = crop_tensor(mask, crop_region)

            item = SEG(cropped_image_tensor, cropped_mask, conf, crop_region, bbox, label, None)
            seg_items.append(item)

        return (image_tensor.shape[1:3], seg_items)

    @staticmethod
    def filter_segs(segs_result, label_filter: str):
        shape, seg_list = segs_result
        if not label_filter or label_filter.strip().lower() == "all":
            return segs_result, (shape, [])

        target_labels = [l.strip() for l in label_filter.split(",")]  # noqa
        logger.debug("检测到: " + str(target_labels))
        target_labels = set(target_labels)

        eye_labels = ["left_eye", "right_eye"]
        brow_labels = ["left_eyebrow", "right_eyebrow"]
        pupil_labels = ["left_pupil", "right_pupil"]

        kept_segs = []
        remained_segs = []

        for x in seg_list:
            match = False
            if x.label in target_labels:
                match = True
            elif "eyes" in target_labels and x.label in eye_labels:
                match = True
            elif "eyebrows" in target_labels and x.label in brow_labels:
                match = True
            elif "pupils" in target_labels and x.label in pupil_labels:
                match = True

            if match:
                kept_segs.append(x)
            else:
                remained_segs.append(x)

        return (shape, kept_segs), (shape, remained_segs)

    def release_device(self):
        del self.bbox_model
        gc.collect()
        if self.yolo_device == "cuda":
            torch.cuda.empty_cache()


class SAMWrapper:
    def __init__(self, model_path, device_mode="AUTO"):
        # print(f"Loading SAM model from: **{os.path.basename(model_path)}**")
        self.model = self._load_model(model_path)
        self.device_mode = device_mode

    def _load_model(self, model_name):
        if "vit_h" in model_name:
            model_kind = "vit_h"
        elif "vit_l" in model_name:
            model_kind = "vit_l"
        else:
            model_kind = "vit_b"
        return sam_model_registry[model_kind](checkpoint=model_name)

    def prepare_device(self):
        # 强制/优先选择 CUDA
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        return self.device

    def release_device(self):
        if self.device_mode == "AUTO":
            self.model.to("cpu")
            torch.cuda.empty_cache()

    def predict_mask(
        self,
        segs,
        image_tensor,
        detection_hint,
        dilation,
        threshold,
        bbox_expansion,
        mask_hint_threshold,
        mask_hint_use_negative,
    ):
        device_used = self.prepare_device()  # noqa

        try:
            img_np = np.clip(255.0 * image_tensor.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
            predictor = SamPredictor(self.model)
            predictor.set_image(img_np, "RGB")
            seg_items = segs[1]
            total_masks = []

            for item in seg_items:
                bbox = item.bbox
                center = center_of_bbox(bbox)
                points = []
                labels = []

                if detection_hint in ["center-1", "mask-point-bbox"]:
                    points.append(center)
                    labels.append(1)

                x1 = max(bbox[0] - bbox_expansion, 0)
                y1 = max(bbox[1] - bbox_expansion, 0)
                x2 = min(bbox[2] + bbox_expansion, img_np.shape[1])
                y2 = min(bbox[3] + bbox_expansion, img_np.shape[0])
                box_input = np.array([x1, y1, x2, y2])

                point_coords = np.array(points) if points else None
                point_labels = np.array(labels) if labels else None

                masks, scores, _ = predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    box=box_input[None, :],
                    multimask_output=True,
                )

                best_idx = np.argmax(scores)
                selected_mask_indices = []
                for idx, score in enumerate(scores):
                    if score >= threshold:
                        selected_mask_indices.append(idx)

                if not selected_mask_indices:
                    selected_mask_indices.append(best_idx)

                for idx in selected_mask_indices:
                    if masks[idx].shape == img_np.shape[:2]:
                        total_masks.append(masks[idx])

            combined_mask = combine_masks(total_masks)

        finally:
            self.release_device()

        if combined_mask is None:
            return torch.zeros((image_tensor.shape[1], image_tensor.shape[2]), dtype=torch.float32)

        if dilation != 0:
            combined_mask_np = combined_mask.numpy().astype(np.uint8)
            combined_mask_np = dilate_mask_array(combined_mask_np, dilation)
            combined_mask = torch.from_numpy(combined_mask_np)

        return combined_mask.float()


class MaskProcessor:
    def __init__(self, yolo_path: str, sam_path: str):

        if not os.path.exists(yolo_path):
            raise FileNotFoundError(f"YOLO model not found at: {yolo_path}")
        if not os.path.exists(sam_path):
            raise FileNotFoundError(f"SAM model not found at: {sam_path}")

        self.bbox_detector = UltraBBoxDetector(yolo_path)
        self.sam_wrapper = SAMWrapper(sam_path)

    def generate_combined_mask(
        self,
        input_image_path: str,
        output_image_path: str,
        detection_threshold: float = 0.5,
        detection_dilation: int = 10,
        sam_threshold: float = 0.93,
        sam_dilation: int = 0,
        filter: str = "all",
    ) -> str:

        if not os.path.exists(input_image_path):
            return ""

        image_tensor = load_image_to_tensor(input_image_path)

        segs = self.bbox_detector.detect(image_tensor, detection_threshold, detection_dilation, 3.0, 1)

        segs_filtered, _ = self.bbox_detector.filter_segs(segs, filter)

        if len(segs_filtered[1]) == 0:
            h, w = image_tensor.shape[1:3]
            empty_mask_tensor = torch.zeros((1, h, w, 3), dtype=torch.float32)
            save_mask_as_image((empty_mask_tensor,), output_image_path)
            return output_image_path

        final_mask = self.sam_wrapper.predict_mask(
            segs_filtered,
            image_tensor,
            detection_hint="center-1",
            dilation=sam_dilation,
            threshold=sam_threshold,
            bbox_expansion=0,
            mask_hint_threshold=0.7,
            mask_hint_use_negative="False",
        )

        visual_mask = mask_to_tensor_visual(final_mask)
        save_mask_as_image((visual_mask,), output_image_path)

        return output_image_path

    def shutdown(self):
        self.bbox_detector.release_device()
