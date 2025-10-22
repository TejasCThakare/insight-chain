"""
Insight-Chain: Dual-Agent Visual Reasoning Demo
Memory-optimized Gradio interface for Google Colab T4 GPU

Usage:
    python demo/app.py
    
Author: Tejas Thakare
GitHub: https://github.com/TejasCThakare/insight-chain
"""

import gradio as gr
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from PIL import Image
from qwen_vl_utils import process_vision_info
import gc
import os

print("🎨 Building Insight-Chain Demo...")

# Memory optimization
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load processor only (models load on-demand to save memory)
processor = AutoProcessor.from_pretrained(
    "models/reasoning_agent/final",
    trust_remote_code=True
)

print("✅ Processor loaded! Models will load on-demand.")


def clear_memory():
    """Aggressively clear GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()


def load_and_run_agent(model_type, image, question):
    """
    Load a model, run inference, and immediately unload it.
    
    Args:
        model_type: "reasoning" or "summary"
        image: PIL Image
        question: Question string
        
    Returns:
        Generated text response
    """
    
    # Clear memory before loading
    clear_memory()
    
    print(f"📥 Loading {model_type} agent...")
    model_path = f"models/{model_type}_agent/final"
    
    # Load with minimal memory usage
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    
    print(f"✅ {model_type.title()} agent loaded")
    print(f"💾 GPU memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    
    # Prepare prompt based on agent type
    if model_type == "reasoning":
        prompt = f"Question: {question}\n\nProvide detailed step-by-step reasoning."
        max_tokens = 256
    else:
        prompt = f"Question: {question}\n\nProvide a concise single-sentence answer."
        max_tokens = 128
    
    # Create conversation
    conversation = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt}
        ]
    }]
    
    # Process inputs
    text_prompt = processor.apply_chat_template(
        conversation, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    image_inputs, video_inputs = process_vision_info(conversation)
    
    inputs = processor(
        text=[text_prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)
    
    print(f"🧠 Generating ({max_tokens} tokens max)...")
    
    # Generate with memory optimization
    with torch.no_grad():
        with torch.cuda.amp.autocast():  # Mixed precision
            output = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.8,
                top_p=0.9,
                repetition_penalty=1.2,
                do_sample=True,
                use_cache=False  # Disable KV cache to save memory
            )
    
    # Decode output
    generated_ids = [
        output_ids[len(input_ids):] 
        for input_ids, output_ids in zip(inputs.input_ids, output)
    ]
    
    result = processor.batch_decode(
        generated_ids, 
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=False
    )[0]
    
    print(f"✅ {model_type.title()} done!")
    
    # Aggressive cleanup
    del model, inputs, output, generated_ids, image_inputs, video_inputs
    clear_memory()
    
    print(f"🗑️ {model_type.title()} unloaded")
    print(f"💾 GPU memory after cleanup: {torch.cuda.memory_allocated()/1024**3:.2f} GB\n")
    
    return result


def analyze_image(image, question):
    """
    Main function to run both agents on an image.
    
    Args:
        image: PIL Image
        question: Question string
        
    Returns:
        Tuple of (reasoning_text, summary_text)
    """
    
    if image is None or not question.strip():
        return "⚠️ Please upload an image and ask a question!", ""
    
    # Resize large images to save memory
    if max(image.size) > 1024:
        image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        print(f"📐 Resized image to {image.size}")
    
    try:
        # Run Agent 1: Reasoning
        print("\n" + "="*70)
        print("🔍 AGENT 1: REASONING")
        print("="*70)
        reasoning_text = load_and_run_agent("reasoning", image, question)
        
        # Run Agent 2: Summary
        print("\n" + "="*70)
        print("📝 AGENT 2: SUMMARY")
        print("="*70)
        summary_text = load_and_run_agent("summary", image, question)
        
        print("\n" + "="*70)
        print("✅ BOTH AGENTS COMPLETE!")
        print("="*70)
        
        return reasoning_text, summary_text
        
    except torch.cuda.OutOfMemoryError as e:
        clear_memory()
        error_msg = (
            "❌ GPU Out of Memory!\n\n"
            "Try:\n"
            "1. Smaller image (< 1024x1024)\n"
            "2. Restart Colab runtime\n"
            "3. Check GPU usage\n\n"
            f"Error: {str(e)}"
        )
        print(error_msg)
        return error_msg, ""
        
    except Exception as e:
        import traceback
        error_msg = f"❌ Error: {str(e)}\n\n{traceback.format_exc()}"
        print(error_msg)
        return error_msg, ""


# Build Gradio Interface
with gr.Blocks(theme=gr.themes.Soft(), title="Insight-Chain") as demo:
    
    # Header
    gr.Markdown("# 🔗 Insight-Chain: Dual-Agent Visual Reasoning")
    
    gr.Markdown("""
    **Explainable AI that shows its reasoning step-by-step**
    
    Upload an image and ask a question. The system uses two specialized agents:
    
    - **Agent 1 (Reasoning):** Generates detailed step-by-step visual analysis
    - **Agent 2 (Summary):** Produces a concise one-sentence answer
    
    ⚡ Optimized for T4 GPU - models load/unload automatically to fit in 15GB VRAM.  
    📸 Large images are automatically resized to 1024x1024.
    """)
    
    # Main interface
    with gr.Row():
        # Left column: Inputs
        with gr.Column():
            image_input = gr.Image(
                type="pil", 
                label="📷 Upload Image",
                height=400
            )
            question_input = gr.Textbox(
                label="❓ Ask a Question",
                placeholder="What's happening in this image?",
                lines=2
            )
            submit_btn = gr.Button(
                "🚀 Analyze Image", 
                variant="primary", 
                size="lg"
            )
        
        # Right column: Outputs
        with gr.Column():
            reasoning_output = gr.Textbox(
                label="🔍 Agent 1: Detailed Reasoning",
                lines=10,
                show_copy_button=True,
                placeholder="Step-by-step reasoning will appear here..."
            )
            summary_output = gr.Textbox(
                label="📝 Agent 2: Concise Summary",
                lines=3,
                show_copy_button=True,
                placeholder="One-sentence summary will appear here..."
            )
    
    # Footer with tips
    gr.Markdown("""
    ---
    
    ### 💡 Tips for Best Results
    
    - Use images under 1024x1024 pixels for optimal performance
    - Processing takes approximately 30-40 seconds for both agents
    - Check the console for detailed memory usage logs
    - Works best with clear, well-lit images
    
    ### 🛠️ Technical Details
    
    - **Base Model:** Qwen2-VL-2B-Instruct (2B parameters)
    - **Fine-tuning:** LoRA adapters (4.3M trainable parameters)
    - **Training Data:** 1043 samples from A-OKVQA + ScienceQA
    - **Hardware:** Google Colab T4 GPU (free tier)
    
    ### 🔗 Links
    
    - **GitHub:** [insight-chain](https://github.com/TejasCThakare/insight-chain)
    - **Author:** Tejas Thakare
    
    Built with ❤️ using Qwen2-VL, Transformers, and Gradio
    """)
    
    # Connect button to function
    submit_btn.click(
        fn=analyze_image,
        inputs=[image_input, question_input],
        outputs=[reasoning_output, summary_output]
    )


# Launch the app
if __name__ == "__main__":
    print("\n" + "="*70)
    print("🚀 Launching Insight-Chain Demo...")
    print("="*70)
    
    demo.queue(max_size=1)  # Process one request at a time
    demo.launch(
        share=True,        # Create public URL
        debug=False,       # Disable debug mode
        show_error=True    # Show errors in UI
    )
