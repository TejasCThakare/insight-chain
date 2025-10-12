"""
Gradio demo for Insight-Chain.
Usage: python demo/app.py
"""

import gradio as gr
import torch
from PIL import Image
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.pipeline import MultiAgentPipeline


# Initialize pipeline
print("🔗 Loading Insight-Chain Pipeline...")

reasoning_model_path = "models/reasoning_agent/final"
summary_model_path = "models/summary_agent/final"

pipeline = MultiAgentPipeline(
    reasoning_model_path=reasoning_model_path,
    summary_model_path=summary_model_path
)

print("✅ Pipeline loaded!")


def process_image(image, question, show_reasoning):
    """Process image and question through pipeline."""
    
    if image is None:
        return "⚠️ Please upload an image."
    
    if not question or question.strip() == "":
        return "⚠️ Please enter a question."
    
    try:
        # Run inference
        result = pipeline.generate(
            image=Image.fromarray(image),
            question=question,
            return_reasoning=show_reasoning
        )
        
        # Format output
        if show_reasoning:
            output = f"## 🧠 Reasoning Chain:\n\n{result['reasoning_chain']}\n\n"
            output += f"## ✅ Final Answer:\n\n{result['final_answer']}"
        else:
            output = f"## ✅ Answer:\n\n{result['final_answer']}"
        
        return output
        
    except Exception as e:
        return f"❌ Error: {str(e)}"


# Create Gradio interface
with gr.Blocks(title="Insight-Chain: Visual Reasoning", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🔗 Insight-Chain: Multi-Agent Visual Reasoning
    
    **Explainable AI that shows its reasoning step-by-step**
    
    Upload an image and ask a question. The system will:
    1. **Reasoning Agent**: Generate detailed step-by-step reasoning
    2. **Summary Agent**: Evaluate reasoning and extract final answer
    
    Built with Qwen2-VL-2B fine-tuned on reasoning chains.
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="numpy", label="Upload Image")
            question_input = gr.Textbox(
                label="Question",
                placeholder="What is happening in this image?",
                lines=2
            )
            show_reasoning = gr.Checkbox(
                label="Show detailed reasoning chain",
                value=True
            )
            submit_btn = gr.Button("🚀 Generate Reasoning", variant="primary")
        
        with gr.Column(scale=1):
            output = gr.Markdown(label="Output")
    
    # Examples
    gr.Examples(
        examples=[
            ["data/examples/example1.jpg", "What activity is shown?"],
            ["data/examples/example2.jpg", "What scientific concept is illustrated?"],
        ],
        inputs=[image_input, question_input],
        label="Example Questions"
    )
    
    # Event handler
    submit_btn.click(
        fn=process_image,
        inputs=[image_input, question_input, show_reasoning],
        outputs=output
    )
    
    gr.Markdown("""
    ---
    **Built with 🧠 by Tejas Thakare** | [GitHub](https://github.com/TejasCThakare/insight-chain)
    """)


if __name__ == "__main__":
    demo.launch(share=True, server_name="0.0.0.0")
