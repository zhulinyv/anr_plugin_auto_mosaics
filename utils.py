import os
import re

import numpy as np
import ujson as json
from PIL import Image

from plugins.anr_plugin_auto_mosaics.detector import detector
from plugins.anr_plugin_auto_mosaics.mosaics import ImageMosaicProcessor
from utils import check_stop, download, read_json, reset_stop
from utils.image_tools import revert_image_info
from utils.images import IMAGE_EXTS, collect_images
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
            except (TypeError, ValueError) as e:
                logger.debug(f"颜色值解析失败, 使用默认色: {e}")
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
            logger.opt(exception=True).debug("处理失败堆栈:")

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

    # 同时输入图片和目录时两者都处理: 先单张图片, 再目录内全部图片。
    # 委托 utils.images.collect_images —— 全项目统一的「哪些算图片 / 怎么排序」实现:
    #   - 只收扩展名受支持的图片 (原先 os.listdir 不过滤, 目录里的 .txt/.md/.db
    #     也会被丢给 PIL 打开, 一张坏文件就把整批任务打断);
    #   - 按自然排序 (frame_2 排在 frame_10 前面, 字典序会反过来);
    #   - 按绝对路径去重 (保留顺序: 先单图, 后目录)。
    images_list = collect_images(mosaic_input_path, mosaic_input_text, exts=IMAGE_EXTS)

    total = len(images_list)
    logger.info(f"开始自动打码, 共 {total} 张图片...")

    # 表情包列表只取一次: 原先写在逐图的 emoji_mosaic 调用里 (os.listdir 每张图都扫一遍目录)。
    # 顺便在这里做一次存在性/非空校验 —— 目录不存在时原先会抛 FileNotFoundError,
    # 被外层捕获成"处理失败", 用户根本看不出是配置问题。
    emoji_files: list[str] = []
    if method == "表情":
        if not emoji or not os.path.isdir(emoji):
            logger.error(f"表情包目录不存在: {emoji}")
            return result_list
        emoji_files = [os.path.join(emoji, i) for i in sorted(os.listdir(emoji))]
        if not emoji_files:
            logger.error(f"表情包目录为空: {emoji}")
            return result_list

    # 逐图隔离异常: 打码是长批处理, 中途一张图损坏 (截断/不是图片/CUDA OOM)
    # 不该让前面已处理的成果白费。原先是整段裸奔, 任何一张抛错都会让 main() 直接退出,
    # result_list 里的结果也一并丢掉 (前端什么都拿不到)。
    failed: list[str] = []
    for image in images_list:
        if check_stop():
            logger.warning("已停止生成!")
            break
        try:
            _process_one = _mosaic_one(
                image, part, method, processor, detector, emoji_files,
                pixel_size, blur_radius, line_width_min, line_width_max,
                line_spacing_min, line_spacing_max, color,
            )
            if _process_one:
                result_list.append(_process_one)
        except Exception as e:
            logger.error(f"处理 {os.path.basename(image)} 失败: {e}")
            logger.opt(exception=True).debug("单图打码失败堆栈:")
            failed.append(os.path.basename(image))

    if failed:
        logger.warning(f"共 {len(failed)} 张处理失败: {', '.join(failed[:5])}")
    return result_list


def _mosaic_one(
    image,
    part,
    method,
    processor,
    detector,
    emoji_files,
    pixel_size,
    blur_radius,
    line_width_min,
    line_width_max,
    line_spacing_min,
    line_spacing_max,
    color,
):
    """处理单张图片, 返回输出路径 (未打码/无输出时返回 None)。异常由调用方隔离。"""
    if is_pure_black_image(image):
        return None

    mask_path = detector(image, part)
    if not mask_path or not os.path.exists(mask_path):
        # 检测没命中 (该图没有需要打码的部位) 时 detector 返回空路径 —— 原先会把空串
        # 传进 mosaic 处理函数, 在那边才炸成"处理失败"。直接跳过更准确。
        logger.debug(f"未检测到需要处理的部位, 跳过: {os.path.basename(image)}")
        return None

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
            emoji_files,
            position="center",
        )
    else:
        logger.error(f"未知的打码方法: {method}")
        return None

    if output_path:
        logger.success(f"处理完成! 图片已保存到 {os.path.abspath(output_path)}")
        logger.debug("正在还原元数据...")
        if revert_image_info(image, output_path):
            logger.success("还原成功!")
        else:
            logger.error("还原失败!")
    return output_path