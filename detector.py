import os
import shutil
import threading

import ujson as json
from PIL import Image, ImageDraw

from utils import read_json
from utils.logger import logger

data = read_json("./plugins/anr_plugin_auto_mosaics/config.json")

# 检测器懒构建: torch / ultralytics / segment_anything 的 import 与模型载入合计 2~5 秒,
# 原来写在模块顶层, 于是每次启动都会替"这次可能根本不用打码"的用户预支掉。
# 现在推迟到第一次真正检测 (或后台预热 warmup()) 时才构建, 且只构建一次。
_build_lock = threading.Lock()
_impl = None


def create_rectangle_mask(image_path: str, coordinates: list):
    original_img = Image.open(image_path)

    img_width, img_height = original_img.size
    mask = Image.new("L", (img_width, img_height), 0)
    draw = ImageDraw.Draw(mask)

    for rect_coords in coordinates:
        x1, y1, x2, y2 = rect_coords
        bbox = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
        draw.rectangle(bbox, fill=255)

    mask.save("./outputs/temp_mask.png")

    return "./outputs/temp_mask.png"


def _build_sam_detector():
    from plugins.anr_plugin_auto_mosaics.sam_detector import MaskProcessor

    logger.debug("加载 YOLO 模型 {}".format(os.path.abspath(data["yolo_model"])))
    logger.debug("加载 SAM 模型 {}".format(os.path.abspath(data["sam_model"])))

    processor = MaskProcessor(data["yolo_model"], data["sam_model"])

    def detect(image_path, part):
        part_mapping = {"欧金金": "penis", "欧芒果": "pussy", "欧派派": "nipple_f"}

        filters = [tag for key, tag in part_mapping.items() if key in part]

        if "欧西利" in part:
            logger.warning("该检测方法不支持该部位(欧西利)检测!")

        _filter = ",".join(filters) if filters else "all"

        output_path = processor.generate_combined_mask(image_path, "./outputs/temp_mask.png", filter=_filter)

        return output_path

    return detect


def _build_nudenet_detector():
    from nudenet import NudeDetector

    logger.debug("加载 NudeNet 检测")
    nude_detector = NudeDetector()

    def detect(image_path: str, part):
        image_last_name = image_path.split(".")[-1]
        shutil.copyfile(image_path, f"./outputs/temp_nudenet.{image_last_name}")
        # 这个库不能包含中文路径

        empty_list = []

        part_list = [
            mapped_value
            for keyword, mapped_value in {
                "欧金金": "MALE_GENITALIA_EXPOSED",
                "欧芒果": "FEMALE_GENITALIA_EXPOSED",
                "欧派派": "EXPOSED_BREAST_F",
                "欧西利": "EXPOSED_ANUS",
            }.items()
            if keyword in part
        ]

        box_list = []
        body = nude_detector.detect(image_path)
        for part in body:
            if part["class"] in part_list:
                empty_list.append(part["class"])

                x1 = part["box"][0]
                y1 = part["box"][1]
                x2 = x1 + part["box"][2]
                y2 = y1 + part["box"][3]
                box_list.append([x1, y1, x2, y2])

        logger.debug(f"检测到: {empty_list}")

        return create_rectangle_mask(image_path, box_list)

    return detect


def _build_yolo_detector():
    from ultralytics import YOLO

    logger.debug("加载 YOLO 模型 {}".format(os.path.abspath(data["yolo_model"])))
    model = YOLO(data["yolo_model"])

    def detect(image_path, part):
        empty_list = []

        part_list = [
            mapped_value
            for keyword, mapped_value in {"欧金金": "penis", "欧芒果": "pussy", "欧派派": "nipple_f"}.items()
            if keyword in part
        ]
        if "欧西利" in part:
            logger.warning("该检测方法不支持该部位(欧西利)检测!")

        box_list = []
        results = model(image_path, verbose=False)
        result = json.loads((results[0]).to_json())
        for i in result:
            if i["name"] in part_list:
                empty_list.append(i["name"])

                x1 = round(i["box"]["x1"])
                y1 = round(i["box"]["y1"])
                x2 = round(i["box"]["x2"])
                y2 = round(i["box"]["y2"])
                box_list.append([x1, y1, x2, y2])

        logger.debug(f"检测到: {empty_list}")

        return create_rectangle_mask(image_path, box_list)

    return detect


def _build_impl():
    """按当前配置选择检测实现 (分支判断与原来完全一致, 只是挪到首次使用时)。"""
    if data["detector"] == "YOLO+SAM" and os.path.exists(data["sam_model"]):
        return _build_sam_detector()

    if data["detector"] == "NudeNet":
        return _build_nudenet_detector()

    if data["detector"] == "YOLO+SAM":
        logger.warning(
            "SAM 模型未下载! 请前往本插件配置设置页面选择 YOLO+SAM 检测方法后选择 SAM 模型并点击保存, 保存后会自动下载模型, 请在看到模型下载完毕提示后再执行关闭或重启操作!"
        )

    return _build_yolo_detector()


def ensure_impl():
    """构建 (或取回) 检测实现; 线程安全, 只构建一次。"""
    global _impl
    if _impl is None:
        with _build_lock:
            if _impl is None:
                _impl = _build_impl()
    return _impl


def detector(image_path, part):
    """检测入口 (保持原有签名): 首次调用时才付 import 与模型载入成本。"""
    return ensure_impl()(image_path, part)


def warmup() -> None:
    """后台预热: 启动时提前把检测器建好, 用户真正使用时不再等待。

    由 utils.plugins 在插件加载完成后于后台线程调用 (失败只记日志)。
    """
    ensure_impl()
