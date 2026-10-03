import os

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy import ndimage

from utils.logger import logger  # 原先缺失: emoji 读取失败时会 NameError 中断整个马赛克


class ImageMosaicProcessor:
    def __init__(self):
        pass

    def _load_images(self, image_path, mask_path):
        try:
            image = Image.open(image_path).convert("RGB")
            mask = Image.open(mask_path).convert("L")
            if mask.size != image.size:
                mask = mask.resize(image.size)
            return image, mask
        except Exception:
            return None, None

    def _save_result(self, image, original_path, method):
        base_name, ext = os.path.splitext(original_path)
        output_path = f"{base_name}_{method}{ext}"
        image.save(output_path)
        return output_path

    def _get_connected_components(self, mask_array):
        binary_mask = (mask_array > 200).astype(int)

        labeled_array, num_features = ndimage.label(binary_mask)

        # ndimage.find_objects 一次遍历就算出全部连通域的包围盒;
        # 原实现用 np.where(labeled_array == i) 为**每个**连通域全图扫描一遍,
        # 蒙版较碎 (数百个连通域) 时是 O(连通域数 × 像素数)。调用方只用 bbox/center,
        # 从不读 "pixels", 所以这里不再保留那份巨大的坐标数组。
        regions = []
        for slice_y, slice_x in ndimage.find_objects(labeled_array):
            if slice_y is None or slice_x is None:
                continue
            min_y, max_y = slice_y.start, slice_y.stop - 1
            min_x, max_x = slice_x.start, slice_x.stop - 1
            regions.append(
                {
                    "bbox": (min_x, min_y, max_x, max_y),
                    "center": ((min_x + max_x) // 2, (min_y + max_y) // 2),
                }
            )

        return regions

    def _calculate_image_brightness(self, image):
        gray_image = image.convert("L")
        gray_array = np.array(gray_image)

        avg_brightness = np.mean(gray_array)
        return avg_brightness

    def pixel_mosaic(self, image_path, mask_path, pixel_size=15):
        image, mask = self._load_images(image_path, mask_path)

        img_array = np.array(image)
        mask_array = np.array(mask)

        height, width = img_array.shape[:2]

        result = img_array.copy()

        mask_indices = mask_array > 200

        # 逐块处理: 掩码通常只覆盖画面一部分, 跳过未命中的块比"全图求块均值再回写"更快
        # (试过 pad+reshape 与 reduceat 两种全向量化写法, 实测在稀疏掩码下都更慢)。
        # 先算好掩码的块级命中图, 避免每块都做一次 np.any 切片。
        pixel_size = max(1, int(pixel_size))
        n_rows = (height + pixel_size - 1) // pixel_size
        n_cols = (width + pixel_size - 1) // pixel_size
        pad_h = n_rows * pixel_size - height
        pad_w = n_cols * pixel_size - width
        padded_mask = np.pad(mask_indices, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=False)
        hit_blocks = padded_mask.reshape(n_rows, pixel_size, n_cols, pixel_size).any(axis=(1, 3))

        for bi, bj in zip(*np.nonzero(hit_blocks)):
            y0, x0 = bi * pixel_size, bj * pixel_size
            y1 = min(y0 + pixel_size, height)
            x1 = min(x0 + pixel_size, width)
            result[y0:y1, x0:x1] = np.mean(img_array[y0:y1, x0:x1], axis=(0, 1)).astype(int)

        result_image = Image.fromarray(result)

        return self._save_result(result_image, image_path, "pixel_mosaic")

    def blur_mosaic(self, image_path, mask_path, blur_radius=12):
        image, mask = self._load_images(image_path, mask_path)

        blurred = image.filter(ImageFilter.GaussianBlur(blur_radius))

        result = image.copy()

        result.paste(blurred, (0, 0), mask)

        return self._save_result(result, image_path, "blur_mosaic")

    def line_mosaic(self, image_path, mask_path, line_width_range=(1, 4), spacing_range=(3, 8)):
        image, mask = self._load_images(image_path, mask_path)

        avg_brightness = self._calculate_image_brightness(image)

        line_color = "white" if avg_brightness < 128 else "black"

        mask_array = np.array(mask)

        regions = self._get_connected_components(mask_array)

        result = image.copy()
        draw = ImageDraw.Draw(result)

        for region_idx, region in enumerate(regions):
            min_x, min_y, max_x, max_y = region["bbox"]

            region_height = max_y - min_y
            region_width = max_x - min_x

            min_side = min(region_height, region_width)
            if min_side < 10:
                continue

            is_horizontal_region = region_width > region_height

            if is_horizontal_region:
                self._draw_vertical_lines(draw, min_x, min_y, max_x, max_y, line_width_range, spacing_range, line_color)
            else:
                self._draw_horizontal_lines(
                    draw, min_x, min_y, max_x, max_y, line_width_range, spacing_range, line_color
                )

        return self._save_result(result, image_path, "line_mosaic")

    def _draw_horizontal_lines(self, draw, min_x, min_y, max_x, max_y, width_range, spacing_range, color):
        region_height = max_y - min_y
        region_width = max_x - min_x

        min_width, max_width = width_range
        min_spacing, max_spacing = spacing_range

        line_count = max(3, int(region_height / (max_spacing + max_width)))
        actual_spacing = region_height / line_count

        base_width = min_width + (max_width - min_width) * (region_height / 500)
        base_width = max(min_width, min(max_width, base_width))

        for i in range(line_count):
            y_pos = min_y + i * actual_spacing + actual_spacing / 2

            if y_pos >= max_y:
                break

            relative_y = (y_pos - min_y) / region_height

            if relative_y < 0.25:
                length_factor = relative_y / 0.25
            elif relative_y < 0.75:
                length_factor = 1.0
            else:
                length_factor = (1.0 - relative_y) / 0.25

            line_length = region_width * length_factor

            start_x = min_x + (region_width - line_length) / 2
            end_x = min_x + (region_width + line_length) / 2

            if relative_y < 0.25:
                width_factor = relative_y / 0.25
            elif relative_y < 0.75:
                width_factor = 1.0
            else:
                width_factor = (1.0 - relative_y) / 0.25

            current_width = max(min_width, min(max_width, base_width * (0.5 + width_factor * 0.5)))

            current_width = max(1, int(current_width))

            draw.line([(start_x, y_pos), (end_x, y_pos)], fill=color, width=current_width)

    def _draw_vertical_lines(self, draw, min_x, min_y, max_x, max_y, width_range, spacing_range, color):
        region_height = max_y - min_y
        region_width = max_x - min_x

        min_width, max_width = width_range
        min_spacing, max_spacing = spacing_range

        line_count = max(3, int(region_width / (max_spacing + max_width)))
        actual_spacing = region_width / line_count

        base_width = min_width + (max_width - min_width) * (region_width / 500)  # 假设500为参考宽度
        base_width = max(min_width, min(max_width, base_width))

        for i in range(line_count):
            x_pos = min_x + i * actual_spacing + actual_spacing / 2

            if x_pos >= max_x:
                break

            relative_x = (x_pos - min_x) / region_width

            if relative_x < 0.25:
                length_factor = relative_x / 0.25
            elif relative_x < 0.75:
                length_factor = 1.0
            else:
                length_factor = (1.0 - relative_x) / 0.25

            line_length = region_height * length_factor

            start_y = min_y + (region_height - line_length) / 2
            end_y = min_y + (region_height + line_length) / 2

            if relative_x < 0.25:
                width_factor = relative_x / 0.25
            elif relative_x < 0.75:
                width_factor = 1.0
            else:
                width_factor = (1.0 - relative_x) / 0.25

            current_width = max(min_width, min(max_width, base_width * (0.5 + width_factor * 0.5)))

            current_width = max(1, int(current_width))

            draw.line([(x_pos, start_y), (x_pos, end_y)], fill=color, width=current_width)

    def line_mosaic_simple(self, image_path, mask_path, line_width=2, line_spacing=5):
        image, mask = self._load_images(image_path, mask_path)

        avg_brightness = self._calculate_image_brightness(image)

        line_color = "white" if avg_brightness < 128 else "black"

        mask_array = np.array(mask)

        regions = self._get_connected_components(mask_array)

        result = image.copy()
        draw = ImageDraw.Draw(result)

        for region_idx, region in enumerate(regions):
            min_x, min_y, max_x, max_y = region["bbox"]

            region_height = max_y - min_y
            region_width = max_x - min_x

            if region_height < 10 or region_width < 10:
                continue

            is_horizontal_region = region_width > region_height

            if is_horizontal_region:
                for x in range(min_x, max_x, line_spacing + line_width):
                    if x < max_x:
                        draw.line([(x, min_y), (x, max_y)], fill=line_color, width=line_width)
            else:
                for y in range(min_y, max_y, line_spacing + line_width):
                    if y < max_y:
                        draw.line([(min_x, y), (max_x, y)], fill=line_color, width=line_width)

        return self._save_result(result, image_path, "line_mosaic_simple")

    def solid_color_mosaic(self, image_path, mask_path, color=(128, 128, 128)):
        image, mask = self._load_images(image_path, mask_path)

        color_layer = Image.new("RGB", image.size, color)

        result = image.copy()

        result.paste(color_layer, (0, 0), mask)

        return self._save_result(result, image_path, "solid_color_mosaic")

    def emoji_mosaic(self, image_path, mask_path, emoji_paths, position="center"):
        image, mask = self._load_images(image_path, mask_path)

        if isinstance(emoji_paths, str):
            emoji_paths = [emoji_paths]

        emojis = []
        for emoji_path in emoji_paths:
            try:
                emoji = Image.open(emoji_path).convert("RGB")
                emojis.append(emoji)
            except Exception as e:
                logger.debug(f"读取表情图片失败, 已跳过: {e}")

        mask_array = np.array(mask)

        regions = self._get_connected_components(mask_array)

        result = image.copy()

        for i, region in enumerate(regions):
            min_x, min_y, max_x, max_y = region["bbox"]

            region_width = max_x - min_x
            region_height = max_y - min_y

            if region_width < 10 or region_height < 10:
                continue

            emoji_idx = i % len(emojis)
            emoji = emojis[emoji_idx]

            emoji_resized = emoji.resize((region_width, region_height))

            paste_x, paste_y = self._calculate_position(
                min_x, min_y, region_width, region_height, emoji_resized.width, emoji_resized.height, position
            )

            result.paste(emoji_resized, (paste_x, paste_y))

        return self._save_result(result, image_path, "emoji_mosaic")

    def _calculate_position(self, min_x, min_y, region_w, region_h, emoji_w, emoji_h, position):
        if callable(position):
            offset_x, offset_y = position(region_w, region_h, emoji_w, emoji_h)
            return min_x + offset_x, min_y + offset_y
        elif position == "center":
            paste_x = min_x + (region_w - emoji_w) // 2
            paste_y = min_y + (region_h - emoji_h) // 2
        elif position == "top-left":
            paste_x = min_x
            paste_y = min_y
        elif position == "top-right":
            paste_x = min_x + region_w - emoji_w
            paste_y = min_y
        elif position == "bottom-left":
            paste_x = min_x
            paste_y = min_y + region_h - emoji_h
        elif position == "bottom-right":
            paste_x = min_x + region_w - emoji_w
            paste_y = min_y + region_h - emoji_h
        else:
            paste_x = min_x
            paste_y = min_y

        return paste_x, paste_y
