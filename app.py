import json
import os
import sys

VENDOR_DIR = os.path.join(os.path.dirname(__file__), "vendor")
if os.path.isdir(VENDOR_DIR) and VENDOR_DIR not in sys.path:
    sys.path.insert(0, VENDOR_DIR)

import gradio as gr
import numpy as np
from PIL import Image

from infer import EnsemblePredictor, make_probability_plot


predictor = EnsemblePredictor()


def _to_pil_image(raw_image, source: str | None = None):
    if raw_image is None:
        return None
    if isinstance(raw_image, Image.Image):
        if source == "手写板" and raw_image.mode in ("RGBA", "LA"):
            alpha = raw_image.getchannel("A")
            return alpha.convert("L")
        return raw_image

    def _array_to_pil(value: np.ndarray) -> Image.Image:
        arr = value
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        if arr.ndim == 3 and arr.shape[-1] == 4:
            if source == "手写板":
                alpha = arr[..., 3]
                if int(alpha.max()) > 0 and int(alpha.min()) < 255:
                    return Image.fromarray(alpha, mode="L")
            rgb = arr[..., :3]
            gray = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.uint8)
            return Image.fromarray(gray, mode="L")
        if arr.ndim == 3 and arr.shape[-1] == 3:
            gray = (0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]).astype(np.uint8)
            return Image.fromarray(gray, mode="L")
        return Image.fromarray(arr)

    if isinstance(raw_image, dict):
        for key in ("composite", "image", "background"):
            value = raw_image.get(key)
            if isinstance(value, Image.Image):
                if source == "手写板" and value.mode in ("RGBA", "LA"):
                    alpha = value.getchannel("A")
                    return alpha.convert("L")
                return value
            if isinstance(value, np.ndarray):
                return _array_to_pil(value)
        layers = raw_image.get("layers")
        if isinstance(layers, list) and layers:
            value = layers[-1]
            if isinstance(value, Image.Image):
                if source == "手写板" and value.mode in ("RGBA", "LA"):
                    alpha = value.getchannel("A")
                    return alpha.convert("L")
                return value
            if isinstance(value, np.ndarray):
                return _array_to_pil(value)
    if isinstance(raw_image, np.ndarray):
        return _array_to_pil(raw_image)
    raise gr.Error("无法解析输入图像，请清空后重试。")


def format_summary(result: dict) -> str:
    return (
        f"预测结果：{result['prediction']}\n"
        f"最高置信度：{result['confidence'] * 100:.2f}%\n"
        f"融合模型数：{result['model_count']}\n"
        f"推理策略：{result['ensemble_desc']}\n"
        f"增强模式：{'开启额外TTA' if result['extra_tta'] else '标准模式'}\n"
        f"推理耗时：{result['latency_ms']:.1f} ms"
    )


def format_top3_html(top3_df) -> str:
    rows = []
    for _, row in top3_df.iterrows():
        rows.append(f"<tr style='border-bottom:1px solid rgba(0,0,0,0.05); transition: background 0.2s;' onmouseover=\"this.style.background='rgba(79,70,229,0.05)'\" onmouseout=\"this.style.background='transparent'\">"
                    f"<td style='padding:12px 10px; font-weight:500;'>{row['Rank']}</td>"
                    f"<td style='padding:12px 10px; font-weight:600; color:#4f46e5;'>{row['Digit']}</td>"
                    f"<td style='padding:12px 10px; color:#475569;'>{row['Confidence']}</td></tr>")
    return (
        "<div style='padding:10px 0'>"
        "<table style='width:100%; border-collapse:collapse; overflow:hidden; border-radius:16px; background:rgba(255,255,255,0.5); box-shadow:0 4px 15px rgba(0,0,0,0.03);'>"
        "<thead><tr style='background: linear-gradient(90deg, #e0e7ff, #fce7f3); color:#374151; font-family:\"Outfit\", sans-serif;'>"
        "<th style='text-align:left; padding:14px 10px; font-weight:600;'>排名</th>"
        "<th style='text-align:left; padding:14px 10px; font-weight:600;'>数字</th>"
        "<th style='text-align:left; padding:14px 10px; font-weight:600;'>置信度</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )


def format_history_html(history: list[dict]) -> str:
    if not history:
        return (
            "<div style='padding:20px; color:#64748b; background:rgba(255,255,255,0.5); border-radius:20px; text-align:center; border: 1px dashed rgba(0,0,0,0.1);'>"
            "✨ 暂无历史记录。完成一次预测后，这里会显示最近结果。"
            "</div>"
        )

    cards = []
    for item in history[-8:][::-1]:
        cards.append(
            "<div style='padding:16px; margin-bottom:12px; background:rgba(255,255,255,0.8); border:1px solid rgba(255,255,255,1); "
            "border-radius:20px; box-shadow:0 8px 24px rgba(0,0,0,0.04); transition: transform 0.2s ease;' "
            "onmouseover=\"this.style.transform='translateY(-2px)'\" onmouseout=\"this.style.transform='translateY(0)'\">"
            f"<div style='display:flex; justify-content:space-between; align-items:center;'><span style='font-size:12px; font-weight:600; color:#818cf8; background:#e0e7ff; padding:4px 8px; border-radius:8px;'>{item['source']}</span>"
            f"<span style='font-size:12px; color:#94a3b8;'>{item['mode']}</span></div>"
            f"<div style='font-size:28px; font-family:\"Outfit\", sans-serif; font-weight:700; color:#1e293b; margin-top:8px;'>预测：<span style='color:#4f46e5;'>{item['digit']}</span></div>"
            f"<div style='font-size:13px; color:#64748b; margin-top:8px; display:flex; justify-content:space-between;'><span>置信度：<strong style='color:#1e293b;'>{item['confidence']}</strong></span> <span>⏱ {item['latency']}</span></div>"
            "</div>"
        )
    return "".join(cards)


def parse_history(history_json: str | None) -> list[dict]:
    if not history_json:
        return []
    try:
        data = json.loads(history_json)
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        return []


def update_history(history: list[dict], source: str, result: dict) -> list[dict]:
    history.append(
        {
            "source": source,
            "mode": "额外TTA" if result["extra_tta"] else "标准模式",
            "digit": result["prediction"],
            "confidence": f"{result['confidence'] * 100:.2f}%",
            "latency": f"{result['latency_ms']:.1f} ms",
        }
    )
    return history[-20:]


def run_prediction(image, use_extra_tta: bool, history_json: str, source: str):
    print(f"[predict] start source={source} extra_tta={use_extra_tta}")
    image = _to_pil_image(image, source=source)
    if image is None:
        raise gr.Error("请先上传手写数字图片，或在手写板中写一个数字。")
    result = predictor.predict(image, use_extra_tta=use_extra_tta)
    summary = format_summary(result)
    plot = make_probability_plot(result["probs"])
    top3_html = format_top3_html(result["top3"])

    history = parse_history(history_json)
    history = update_history(history, source, result)
    history_html = format_history_html(history)
    print(f"[predict] done pred={result['prediction']} conf={result['confidence']:.4f} latency_ms={result['latency_ms']:.1f}")
    return (
        summary,
        top3_html,
        plot,
        result["processed_image"],
        result["mask_image"],
        json.dumps(history, ensure_ascii=False),
        history_html,
    )


def clear_history():
    return "[]", format_history_html([])


CSS = """
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&family=Inter:wght@400;500;600&display=swap');

:root {
  --bg-main: #f3f4f6;
  --bg-panel: rgba(255, 255, 255, 0.65);
  --line-soft: rgba(255, 255, 255, 0.5);
  --text-main: #1e293b;
  --text-soft: #475569;
  --accent: #4f46e5;
  --accent-hover: #4338ca;
  --accent-2: #ec4899;
  --shadow: 0 10px 40px -10px rgba(0,0,0,0.08);
  --shadow-hover: 0 20px 40px -10px rgba(79, 70, 229, 0.15);
  --glass-border: 1px solid rgba(255, 255, 255, 0.8);
}

body, .gradio-container {
  background: 
    radial-gradient(circle at 15% 50%, rgba(79, 70, 229, 0.08), transparent 25%),
    radial-gradient(circle at 85% 30%, rgba(236, 72, 153, 0.08), transparent 25%),
    linear-gradient(135deg, #e0e7ff 0%, #fce7f3 100%);
  color: var(--text-main);
  font-family: 'Inter', "PingFang SC", sans-serif;
  min-height: 100vh;
}

.app-shell {
  max-width: 1200px;
  margin: 0 auto;
  padding: 40px 16px;
  animation: fadeIn 0.8s ease-out;
}

.hero {
  background: linear-gradient(135deg, rgba(255,255,255,0.9), rgba(255,255,255,0.5));
  border: var(--glass-border);
  border-radius: 32px;
  padding: 40px;
  box-shadow: var(--shadow);
  backdrop-filter: blur(20px);
  -webkit-backdrop-filter: blur(20px);
  margin-bottom: 24px;
  transition: transform 0.3s ease, box-shadow 0.3s ease;
  position: relative;
  overflow: hidden;
}
.hero::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0; height: 4px;
  background: linear-gradient(90deg, var(--accent), var(--accent-2));
}
.hero:hover {
  transform: translateY(-2px);
  box-shadow: var(--shadow-hover);
}

.hero h1 { 
  margin: 0 0 16px; 
  font-size: 42px; 
  font-family: 'Outfit', sans-serif;
  font-weight: 700;
  background: linear-gradient(90deg, var(--text-main), var(--accent));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  letter-spacing: -0.02em; 
}

.hero p { 
  margin: 0; 
  color: var(--text-soft); 
  font-size: 16px; 
  line-height: 1.8; 
  max-width: 800px; 
}

.badge-row { 
  display: flex; 
  gap: 12px; 
  flex-wrap: wrap; 
  margin-top: 24px; 
}

.badge {
  background: rgba(255, 255, 255, 0.8);
  border: 1px solid rgba(79, 70, 229, 0.2);
  border-radius: 999px;
  padding: 8px 16px;
  font-size: 13px;
  font-weight: 500;
  color: var(--accent);
  box-shadow: 0 2px 10px rgba(0,0,0,0.02);
  transition: all 0.2s ease;
}
.badge:hover {
  background: var(--accent);
  color: white;
  transform: translateY(-1px);
}

.panel {
  background: var(--bg-panel);
  border: var(--glass-border);
  border-radius: 28px;
  box-shadow: var(--shadow);
  backdrop-filter: blur(20px);
  -webkit-backdrop-filter: blur(20px);
  padding: 24px;
  transition: transform 0.3s ease, box-shadow 0.3s ease;
  margin-bottom: 16px;
}
.panel:hover {
  box-shadow: var(--shadow-hover);
}

.section-copy { 
  color: var(--text-soft); 
  font-size: 15px; 
  line-height: 1.8; 
}

.gr-button-primary { 
  background: linear-gradient(135deg, var(--accent), #6366f1) !important; 
  border: none !important; 
  color: white !important;
  font-weight: 600 !important;
  border-radius: 14px !important;
  transition: all 0.3s ease !important;
  box-shadow: 0 4px 15px rgba(79, 70, 229, 0.3) !important;
}
.gr-button-primary:hover {
  transform: translateY(-2px) scale(1.02) !important;
  box-shadow: 0 8px 25px rgba(79, 70, 229, 0.4) !important;
}

.gr-button-secondary { 
  background: rgba(255,255,255,0.8) !important;
  border: 1px solid rgba(0,0,0,0.05) !important; 
  color: var(--text-main) !important;
  border-radius: 14px !important;
  font-weight: 500 !important;
  transition: all 0.3s ease !important;
}
.gr-button-secondary:hover {
  background: white !important;
  box-shadow: 0 4px 12px rgba(0,0,0,0.05) !important;
  transform: translateY(-1px) !important;
}

@keyframes fadeIn {
  from { opacity: 0; transform: translateY(10px); }
  to { opacity: 1; transform: translateY(0); }
}

footer { display:none !important; }
"""


with gr.Blocks(css=CSS, title="CNN 手写数字识别系统") as demo:
    with gr.Column(elem_classes=["app-shell"]):
        history_state = gr.Textbox(value="[]", visible=False)
        upload_source = gr.Textbox(value="上传图片", visible=False)
        sketch_source = gr.Textbox(value="手写板", visible=False)

        gr.Markdown(
            """
            <div style="text-align: center; margin: 10px 0 30px 0;">
                <h1 style="font-size: 56px; font-weight: 800; font-family: 'Outfit', sans-serif; background: linear-gradient(90deg, #4f46e5, #ec4899); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0; padding-bottom: 10px;">
                    ✨ 智能数字识别系统
                </h1>
                <p style="font-size: 18px; color: #64748b; margin-top: 5px; font-family: 'Inter', sans-serif;">
                    Powered by Premium Vision & High-Accuracy Residual Ensembles
                </p>
            </div>
            """
        )



        with gr.Accordion("📖 使用说明与技巧 (点击展开)", open=False, elem_classes=["panel"]):
            gr.Markdown(
                "<div class='section-copy'>"
                "• 建议上传单个数字，黑底白字或白底黑字皆可，系统会自动完成前景提取、裁剪、居中和 28×28 归一化。<br>"
                "• 手写板请尽量把数字写在画布中央，保持笔画连续和完整。<br>"
                "• <b>什么是 TTA？</b> 开启额外 TTA 会使用更多的平移视角进行融合预测，能有效提升模型的稳健性，但会稍微增加一点推理时间。"
                "</div>"
            )

        with gr.Row():
            tta_toggle = gr.Checkbox(label="⚙️ 开启额外TTA（增强预测稳健性）", value=False)

        with gr.Tabs():
            with gr.Tab("📁 上传图片识别"):
                with gr.Row():
                    with gr.Column(scale=6, elem_classes=["panel"]):
                        upload_input = gr.Image(type="pil", image_mode="L", label="上传手写数字图片", sources=["upload"], height=420)
                        upload_button = gr.Button("🚀 立即识别上传图片", variant="primary", size="lg")
                    with gr.Column(scale=5, elem_classes=["panel"]):
                        gr.Markdown("### 📊 识别结果分析")
                        upload_summary = gr.Textbox(label="基本信息", lines=4)
                        upload_top3 = gr.HTML(label="Top-3 概率排名")
                        upload_plot = gr.Plot(label="概率分布图")
                        with gr.Accordion("🔍 查看预处理中间结果 (折叠)", open=False):
                            with gr.Row():
                                upload_processed = gr.Image(type="pil", image_mode="L", label="最终模型输入 (28x28)")
                                upload_mask = gr.Image(type="pil", image_mode="L", label="前景提取掩码")
                            gr.Markdown(
                                "<div class='section-copy'>"
                                "说明：左侧图像是最终喂给模型的 28×28 灰度图。若数字缺笔画或偏移严重，建议重传后再预测。"
                                "</div>"
                            )

            with gr.Tab("✨ 手写板识别"):
                with gr.Row():
                    with gr.Column(scale=6, elem_classes=["panel"]):
                        sketch_input = gr.Sketchpad(type="pil", image_mode="RGBA", label="在下方手写板中写一个数字", height=520)
                        sketch_button = gr.Button("🚀 立即识别手写内容", variant="primary", size="lg")
                    with gr.Column(scale=5, elem_classes=["panel"]):
                        gr.Markdown("### 📊 识别结果分析")
                        sketch_summary = gr.Textbox(label="基本信息", lines=4)
                        sketch_top3 = gr.HTML(label="Top-3 概率排名")
                        sketch_plot = gr.Plot(label="概率分布图")
                        with gr.Accordion("🔍 查看预处理中间结果 (折叠)", open=False):
                            with gr.Row():
                                sketch_processed = gr.Image(type="pil", image_mode="L", label="最终模型输入 (28x28)")
                                sketch_mask = gr.Image(type="pil", image_mode="L", label="前景提取掩码")

        with gr.Row(elem_classes=["panel"]):
            with gr.Column(scale=4):
                gr.Markdown("### 📜 最近历史记录")
            with gr.Column(scale=1):
                clear_history_button = gr.Button("🗑️ 清空历史记录", variant="secondary")
        
        with gr.Row():
            with gr.Column():
                history_html = gr.HTML(format_history_html([]))

        upload_button.click(
            fn=run_prediction,
            inputs=[upload_input, tta_toggle, history_state, upload_source],
            outputs=[upload_summary, upload_top3, upload_plot, upload_processed, upload_mask, history_state, history_html],
            queue=False,
        )
        sketch_button.click(
            fn=run_prediction,
            inputs=[sketch_input, tta_toggle, history_state, sketch_source],
            outputs=[sketch_summary, sketch_top3, sketch_plot, sketch_processed, sketch_mask, history_state, history_html],
            queue=False,
        )
        clear_history_button.click(fn=clear_history, inputs=None, outputs=[history_state, history_html], queue=False)


if __name__ == "__main__":
    server_name = "0.0.0.0" if os.getenv("PORT") else "127.0.0.1"
    server_port = int(os.getenv("PORT", "7860"))
    try:
        demo.launch(server_name=server_name, server_port=server_port, share=False, show_error=True)
    except ValueError as e:
        if "localhost is not accessible" in str(e):
            try:
                demo.launch(server_name=server_name, server_port=server_port, share=True, show_error=True)
            except OSError as oe:
                if "Cannot find empty port" in str(oe):
                    demo.launch(server_name=server_name, server_port=None, share=True, show_error=True)
                else:
                    raise
        else:
            raise
    except OSError as e:
        if "Cannot find empty port" in str(e):
            demo.launch(server_name=server_name, server_port=None, share=False, show_error=True)
        else:
            raise
