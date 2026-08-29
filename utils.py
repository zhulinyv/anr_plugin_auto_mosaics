import os
import re

import numpy as np
import ujson as json
from PIL import Image

from plugins.anr_plugin_auto_mosaics.detector import detector
from plugins.anr_plugin_auto_mosaics.mosaics import ImageMosaicProcessor
from utils import check_stop, download, read_json, reset_stop
from utils.image_tools import revert_image_info
from utils.logger import logger


def color_change(color, default=(0, 0, 0)):
    """把颜色字符串解析成 (r, g, b) 元组。

    支持 #RRGGBB / #RGB / rgb(r, g, b) / rgba(...) / "r, g, b" / "r g b" 等格式,
    无法解析时回退到 default (黑色)。
    """
    if not isinstance(color, str):
        # 直接传入 (r, g, b) 元组/列表时原样规整
        if isinstance(color, (tuple, list)) and len(color) >= 3:
            try:
                return tuple(max(0, min(255, round(float(c)))) for c in color[:3])
            except (TypeError, ValueError):
                pass
        return default

    color = color.strip()

    # 十六进制: #RRGGBB / RRGGBB / #RGB
    m = re.fullmatch(r"#?([0-9a-fA-F]{6})", color)
    if m:
        h = m.group(1)
        return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))
    m = re.fullmatch(r"#?([0-9a-fA-F]{3})", color)
    if m:
        return tuple(int(c * 2, 16) for c in m.group(1))

    # 数字列表: rgb(255, 0, 0) / rgba(255, 0, 0, 1) / "255, 0, 0" / "255 0 0"
    numbers = re.findall(r"[\d.]+", color)
    if len(numbers) >= 3:
        try:
            values = tuple(round(float(n)) for n in numbers[:3])
        except ValueError:
            return default
        return tuple(max(0, min(255, v)) for v in values)

    return default


def save_config(detector, yolo_model, sam_model: str):
    """保存打码插件配置, 若 SAM 模型缺失则自动下载。"""
    data = read_json("./plugins/anr_plugin_auto_mosaics/config.json")
    data["detector"] = detector
    data["yolo_model"] = yolo_model
    data["sam_model"] = sam_model

    with open("./plugins/anr_plugin_auto_mosaics/config.json", "w", encoding="utf-8") as file:
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

    return "配置已保存, 即时生效!"


def is_pure_black_image(image_path):
    img = Image.open(image_path)
    img_array = np.array(img)
    if img.mode == "RGBA":
        rgb_array = img_array[:, :, :3]
        return bool(np.all(rgb_array == 0))
    return bool(np.all(img_array == 0))


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
    reset_stop()  # 重置本任务的停止信号

    result_list = []

    # 同时输入图片和目录时两者都处理: 先单张图片, 再目录内全部图片
    images_list = []
    if mosaic_input_text:
        images_list.append(mosaic_input_text)
    if mosaic_input_path:
        images_list.extend(
            mosaic_input_path + f"/{i}" for i in os.listdir(mosaic_input_path)
        )
    # 去重 (保留顺序: 先图片, 再目录)
    seen = set()
    unique_list = []
    for img in images_list:
        key = os.path.abspath(img)
        if key not in seen:
            seen.add(key)
            unique_list.append(img)
    images_list = unique_list

    total = len(images_list)
    logger.info(f"开始自动打码, 共 {total} 张图片...")
    for image in images_list:
        if check_stop():
            logger.warning("已停止生成!")
            break

        if is_pure_black_image(image):
            continue

        mask_path = detector(image, part)
        output_path = None
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
        else:
            logger.error(f"未知的打码方法: {method}")

        if output_path:
            logger.success(f"处理完成! 图片已保存到 {os.path.abspath(output_path)}")
            result_list.append(output_path)

            logger.debug("正在还原元数据...")
            if revert_image_info(image, output_path):
                logger.success("还原成功!")
            else:
                logger.error("还原失败!")

    return result_list