import os
import shutil

import ujson as json
from PIL import Image, ImageDraw

from utils import read_json
from utils.logger import logger

data = read_json("./plugins/anr_plugin_auto_mosaics/config.json")


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


if data["detector"] == "YOLO+SAM" and os.path.exists(data["sam_model"]):
    from plugins.anr_plugin_auto_mosaics.sam_detector import MaskProcessor

    logger.debug("加载 YOLO 模型 {}".format(os.path.abspath(data["yolo_model"])))
    logger.debug("加载 SAM 模型 {}".format(os.path.abspath(data["sam_model"])))

    processor = MaskProcessor(data["yolo_model"], data["sam_model"])

    def detector(image_path, part):
        part_mapping = {"欧金金": "penis", "欧芒果": "pussy", "欧派派": "nipple_f"}

        filters = [tag for key, tag in part_mapping.items() if key in part]

        if "欧西利" in part:
            logger.warning("该检测方法不支持该部位(欧西利)检测!")

        _filter = ",".join(filters) if filters else "all"

        output_path = processor.generate_combined_mask(image_path, "./outputs/temp_mask.png", filter=_filter)

        return output_path

elif data["detector"] == "NudeNet":
    from nudenet import NudeDetector

    logger.debug("加载 NudeNet 检测")
    nude_detector = NudeDetector()

    def detector(image_path: str, part):
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

else:
    if data["detector"] == "YOLO+SAM":
        logger.warning(
            "SAM 模型未下载! 请前往本插件配置设置页面选择 YOLO+SAM 检测方法后选择 SAM 模型并点击保存, 保存后会自动下载模型, 请在看到模型下载完毕提示后再执行关闭或重启操作!"
        )

    from ultralytics import YOLO

    logger.debug("加载 YOLO 模型 {}".format(os.path.abspath(data["yolo_model"])))
    model = YOLO(data["yolo_model"])

    def detector(image_path, part):
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
