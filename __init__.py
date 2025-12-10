import gradio as gr

from plugins.anr_plugin_auto_mosaics.utils import (
    main,
    return_method_visible,
    return_model_visible,
    save_config,
)
from utils import read_json, stop_generate, tk_asksavefile_asy
from utils.image_tools import return_array_image


def plugin():
    data = read_json("./plugins/anr_plugin_auto_mosaics/config.json")

    with gr.Tab("自动打码"):
        with gr.Tab("批量处理"):
            with gr.Row():
                mosaic_input_path = gr.Textbox(label="批处理路径(同时输入路径和图片时仅处理图片)")
                part = gr.CheckboxGroup(
                    ["欧金金", "欧芒果", "欧派派", "欧西利"],
                    value=["欧金金", "欧芒果"],
                    label="处理部位",
                    interactive=True,
                )
            with gr.Row():
                method = gr.Radio(
                    ["像素", "模糊", "线条", "纯色", "表情"], value="像素", label="打码方法", interactive=True
                )
                pixel_size = gr.Slider(1, 100, 15, step=1, label="像素大小", visible=True, interactive=True)
                blur_radius = gr.Slider(1, 100, 12, step=1, label="模糊半径", visible=False, interactive=True)
                emoji = gr.Textbox(
                    "./plugins/anr_plugin_auto_mosaics/emoji", label="表情目录", visible=False, interactive=True
                )
            color = gr.ColorPicker(label="填充颜色", visible=False, interactive=True)
            line_width_min = gr.Slider(1, 20, 3, step=1, label="最小线条宽度", visible=False, interactive=True)
            line_width_max = gr.Slider(2, 20, 10, step=1, label="最大线条宽度", visible=False, interactive=True)
            line_spacing_min = gr.Slider(1, 30, 10, step=1, label="最小线条间隔", visible=False, interactive=True)
            line_spacing_max = gr.Slider(1, 30, 15, step=1, label="最大线条间隔", visible=False, interactive=True)
            method.change(
                return_method_visible,
                method,
                outputs=[
                    pixel_size,
                    blur_radius,
                    line_spacing_max,
                    line_spacing_min,
                    line_width_max,
                    line_width_min,
                    color,
                    emoji,
                ],
            )

            with gr.Row():
                with gr.Column():
                    mosaic_input_image = gr.Image(type="numpy", interactive=False, label="Input")
                    with gr.Row():
                        mosaic_input_text = gr.Textbox(visible=False)
                        mosaic_input_btn = gr.Button("选择图片")
                        mosaic_clear_btn = gr.Button("清除选择")
                mosaic_clear_btn.click(lambda x: x, gr.Textbox(None, visible=False), mosaic_input_text)
                mosaic_input_btn.click(tk_asksavefile_asy, inputs=[], outputs=[mosaic_input_text])
                mosaic_input_text.change(return_array_image, mosaic_input_text, mosaic_input_image)
                with gr.Column():
                    with gr.Row():
                        mosaic_button = gr.Button("开始处理")
                        mosaic_stop = gr.Button("停止处理")
                    outputs = gr.Gallery(interactive=False, label="Output")
            mosaic_button.click(
                main,
                inputs=[
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
                ],
                outputs=outputs,
            )
            mosaic_stop.click(stop_generate)
        with gr.Tab("配置设置"):
            save_button = gr.Button("保存")
            output_info = gr.Textbox(label="输出信息", visible=False)
            detector = gr.Radio(
                ["YOLO+SAM", "YOLO", "NudeNet"], value=data["detector"], label="检测方法", interactive=True
            )
            yolo_model = gr.Textbox(
                data["yolo_model"],
                label="YOLO 模型",
                visible=True if "YOLO" in data["detector"] else False,
            )
            sam_model = gr.Dropdown(
                [
                    "./plugins/anr_plugin_auto_mosaics/models/sams/sam_vit_b_01ec64.pth",
                    "./plugins/anr_plugin_auto_mosaics/models/sams/sam_vit_l_0b3195.pth",
                    "./plugins/anr_plugin_auto_mosaics/models/sams/sam_vit_h_4b8939.pth",
                ],
                value=data["sam_model"],
                label="SAM 模型",
                visible=True if "SAM" in data["detector"] else False,
            )

            detector.change(return_model_visible, detector, outputs=[yolo_model, sam_model])
            save_button.click(save_config, inputs=[detector, yolo_model, sam_model], outputs=output_info)
