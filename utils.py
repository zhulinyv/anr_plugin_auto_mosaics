import os
import re

import gradio as gr
import numpy as np
import ujson as json
from PIL import Image

from plugins.anr_plugin_auto_mosaics.detector import detector
from plugins.anr_plugin_auto_mosaics.mosaics import ImageMosaicProcessor
from utils import download, read_json
from utils.image_tools import revert_image_info
from utils.logger import logger


def color_change(color):
    numbers = re.findall(r"[\d.]+", color)

    r = float(numbers[0])
    g = float(numbers[1])
    b = float(numbers[2])

    return (round(r), round(g), round(b))


def return_model_visible(detector):
    if detector == "YOLO+SAM":
        return gr.update(visible=True), gr.update(visible=True)
    elif detector == "YOLO":
        return gr.update(visible=True), gr.update(visible=False)
    else:
        return gr.update(visible=False), gr.update(visible=False)


def save_config(detector, yolo_model, sam_model: str):
    data = read_json("./plugins/anr_plugin_auto_mosaics/config.json")
    data["detector"] = detector
    data["yolo_model"] = yolo_model
    data["sam_model"] = sam_model

    with open(
        "./plugins/anr_plugin_auto_mosaics/config.json", "w", encoding="utf-8"
    ) as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

    if detector == "YOLO+SAM" and not os.path.exists(data["sam_model"]):
        sam_model = sam_model.split("/")[-1]
        logger.warning(f"本地未发现 {sam_model} 模型!")
        logger.info(f"正在下载 {sam_model} 模型...")
        try:
            download(
                f"https://huggingface.co/datasets/Xytpz/SAM_Models/resolve/main/{sam_model}?download=true",
                f"./plugins/anr_plugin_auto_mosaics/models/sams/{sam_model}",
            )
            logger.success(f"{sam_model} 模型下载完成!")
        except Exception as e:
            logger.error(f"出现错误! {e}")

    return gr.update(value="配置已保存, 重启后生效!", visible=True)


def return_method_visible(method):
    if method == "像素":
        return (
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
        )
    elif method == "模糊":
        return (
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
        )
    elif method == "线条":
        return (
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
        )
    elif method == "纯色":
        return (
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=False),
        )
    elif method == "表情":
        return (
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=True),
        )


def is_pure_black_image(image_path):
    img = Image.open(image_path)
    img_array = np.array(img)

    if img.mode == "RGBA":
        rgb_array = img_array[:, :, :3]
        return np.all(rgb_array == 0)
    else:
        return np.all(img_array == 0)


processor = ImageMosaicProcessor()


def main(
    method,
    pixel_size,
    blur_radius,
    line_spacing_max,
    line_spacing_min,
    line_width_max,
    line_width_min,
    color,
    emoji,
    mosaic_input_path,
    mosaic_input_text,
    part,
):
    with open("./outputs/temp_break.json", "w") as f:
        json.dump({"break": False}, f)

    result_list = []

    if mosaic_input_text:
        images_list = [mosaic_input_text]
    else:
        images_list = [
            mosaic_input_path + f"/{i}" for i in os.listdir(mosaic_input_path)
        ]

    for image in images_list:
        _break = read_json("./outputs/temp_break.json")
        if _break["break"]:
            logger.warning("已停止生成!")
            break

        if is_pure_black_image(image):
            continue

        mask_path = detector(image, part)
        if method == "像素":
            output_path = processor.pixel_mosaic(
                image, mask_path, pixel_size=pixel_size
            )
        elif method == "模糊":
            output_path = processor.blur_mosaic(
                image, mask_path, blur_radius=blur_radius
            )
        elif method == "线条":
            output_path = processor.line_mosaic(
                image,
                mask_path,
                line_width_range=(line_width_min, line_width_max),
                spacing_range=(line_spacing_min, line_spacing_max),
            )
        elif method == "纯色":
            output_path = processor.solid_color_mosaic(
                image, mask_path, color=color_change(color)
            )
        elif method == "表情":
            output_path = processor.emoji_mosaic(
                image,
                mask_path,
                [emoji + f"/{i}" for i in os.listdir(emoji)],
                position="center",
            )

        if output_path:
            logger.success(f"处理完成! 图片已保存到 {os.path.abspath(output_path)}")
            result_list.append(output_path)

            logger.debug("正在还原元数据...")
            if revert_image_info(image, output_path):
                logger.success("还原成功!")
            else:
                logger.error("还原失败!")

    return result_list
