"""自动打码插件: 基于 YOLO / SAM / NudeNet 的批量打码。"""
from __future__ import annotations

import ujson as json

from plugins.anr_plugin_auto_mosaics.utils import main, save_config
from utils.helpers import read_json
from utils.plugins import Action, Field, Panel, Plugin


def _run_main(values: dict) -> dict:
    """把表单值映射到 main() 并返回处理结果 (展示生成的图片 + 提示信息)。"""
    result = main(
        method=values.get("method", "像素"),
        pixel_size=int(values.get("pixel_size", 15)),
        blur_radius=int(values.get("blur_radius", 12)),
        line_spacing_max=int(values.get("line_spacing_max", 15)),
        line_spacing_min=int(values.get("line_spacing_min", 10)),
        line_width_max=int(values.get("line_width_max", 10)),
        line_width_min=int(values.get("line_width_min", 3)),
        color=values.get("color", "#000000"),
        emoji=values.get("emoji", "./plugins/anr_plugin_auto_mosaics/emoji"),
        mosaic_input_path=values.get("path", ""),
        mosaic_input_text=values.get("image", ""),
        part=values.get("part", ["欧金金", "欧芒果"]),
    )
    return {"images": result, "message": f"处理完成! 共生成 {len(result)} 张图片"}


def register(plugin: Plugin):
    # 配置默认值
    try:
        data = read_json("./plugins/anr_plugin_auto_mosaics/config.json")
    except Exception:
        data = {}

    process_panel = Panel(
        id="process",
        title="批量处理",
        icon="🛠️",
        fields=[
            Field(id="path", label="批处理路径", type="path", folder=True, file=False),
            Field(id="image", label="或上传单张图片", type="image"),
            Field(id="part", label="处理部位", type="checkbox_group", options=["欧金金", "欧芒果", "欧派派", "欧西利"], default=["欧金金", "欧芒果"]),
            Field(id="method", label="打码方法", type="radio", options=["像素", "模糊", "线条", "纯色", "表情"], default="像素"),
            Field(id="pixel_size", label="像素大小", type="slider", min=1, max=100, step=1, default=15, show_if={"field": "method", "equals": "像素"}),
            Field(id="blur_radius", label="模糊半径", type="slider", min=1, max=100, step=1, default=12, show_if={"field": "method", "equals": "模糊"}),
            Field(id="line_width_min", label="最小线条宽度", type="slider", min=1, max=20, step=1, default=3, show_if={"field": "method", "equals": "线条"}),
            Field(id="line_width_max", label="最大线条宽度", type="slider", min=2, max=20, step=1, default=10, show_if={"field": "method", "equals": "线条"}),
            Field(id="line_spacing_min", label="最小线条间隔", type="slider", min=1, max=30, step=1, default=10, show_if={"field": "method", "equals": "线条"}),
            Field(id="line_spacing_max", label="最大线条间隔", type="slider", min=1, max=30, step=1, default=15, show_if={"field": "method", "equals": "线条"}),
            Field(id="color", label="填充颜色", type="color", default="#000000", show_if={"field": "method", "equals": "纯色"}),
            Field(id="emoji", label="表情目录", type="path", default="./plugins/anr_plugin_auto_mosaics/emoji", folder=True, file=False, show_if={"field": "method", "equals": "表情"}),
        ],
        actions=[
            Action(id="run", label="🛠️ 开始处理", uses_novelai=False, inputs=["method", "pixel_size", "blur_radius", "line_spacing_max", "line_spacing_min", "line_width_max", "line_width_min", "color", "emoji", "path", "image", "part"], handler=_run_main),
        ],
    )

    config_panel = Panel(
        id="config",
        title="配置设置",
        icon="⚙️",
        show_output=False,
        fields=[
            Field(id="detector", label="检测方法", type="radio", options=["YOLO+SAM", "YOLO", "NudeNet"], default=data.get("detector", "YOLO+SAM")),
            Field(id="yolo_model", label="YOLO 模型", type="text", default=data.get("yolo_model", ""), show_if={"field": "detector", "contains": "YOLO"}),
            Field(id="sam_model", label="SAM 模型", type="select", options=[
                "./plugins/anr_plugin_auto_mosaics/models/sams/sam_vit_b_01ec64.pth",
                "./plugins/anr_plugin_auto_mosaics/models/sams/sam_vit_l_0b3195.pth",
                "./plugins/anr_plugin_auto_mosaics/models/sams/sam_vit_h_4b8939.pth",
            ], default=data.get("sam_model", "./plugins/anr_plugin_auto_mosaics/models/sams/sam_vit_b_01ec64.pth"), show_if={"field": "detector", "contains": "SAM"}),
        ],
        actions=[
            Action(id="save", label="💾 保存配置", uses_novelai=False, inputs=["detector", "yolo_model", "sam_model"], handler=lambda v: {"text": save_config(v.get("detector", "YOLO+SAM"), v.get("yolo_model", ""), v.get("sam_model", ""))}),
        ],
    )

    plugin.title = "自动打码"
    plugin.description = "批量检测并打码敏感部位 (YOLO / SAM / NudeNet)"
    plugin.icon = "🫧"
    plugin.panels.extend([process_panel, config_panel])