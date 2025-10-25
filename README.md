# 🔗 Insight-Chain: Dual-Agent Visual Reasoning

**Tejas Thakare**

A visual question answering system using dual LoRA adapters on Qwen2-VL-2B for transparent, step-by-step reasoning.

⭐ **If you find this project useful, please give us a star on GitHub!**

---

## 🚀 Updates

- **[Oct 2025]** Demo deployment complete with public Gradio interface
- **[Oct 2025]** Training and inference code now available
- **[Oct 2025]** Project repository is live!

---

## 🔍 Overview

Given an image and question, Insight-Chain generates both detailed reasoning steps and a concise summary. The system uses two specialized LoRA adapters fine-tuned on Qwen2-VL-2B: one for generating multi-step visual analysis, and another for producing condensed answers. Both adapters are loaded sequentially to minimize memory usage (~10GB peak).

**Key Features:**
- ✨ Explainable AI with transparent reasoning
- 🧠 Memory-efficient sequential loading (37% reduction vs standard)
- 🔄 Complete end-to-end pipeline from training to deployment
- 💰 Zero-cost training on Google Colab

---

## 🛠️ Quick Start

### Installation
```bash
git clone https://github.com/TejasCThakare/insight-chain.git
cd insight-chain
pip install -r requirements.txt
```

### Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.46+
- 15GB GPU (or Google Colab T4)

---

## 📊 Complete Pipeline: Data → Training → Inference → Demo

### Step 1: Data Preparation
```bash
python scripts/run_data_pipeline.py
```

**What it does:**
- Downloads A-OKVQA and ScienceQA datasets
- Converts to unified format for dual-agent training
- Creates train/validation splits
- Saves to `data/processed/`

**Output:**
- `data/processed/train.json`
- `data/processed/val.json`
- `data/processed/images/`

---

### Step 2: Train Reasoning Agent
```bash
python scripts/run_training.py --mode reasoning
```

**Training details:**
- Learning Rate: `1e-4`
- LoRA Rank: `16`, Alpha: `32`
- Batch Size: `1` (gradient accumulation: `4`)
- Time: ~20 minutes (T4 GPU)

**Output:** `models/reasoning_agent/final/`

---

### Step 3: Train Summary Agent
```bash
python scripts/run_training.py --mode summary
```

**Training details:**
- Same configuration as reasoning agent
- Time: ~20 minutes (T4 GPU)

**Output:** `models/summary_agent/final/`

---

### Step 4: Run Inference

**Test Reasoning Agent:**
```bash
python scripts/run_inference.py \
    --model "models/reasoning_agent/final" \
    --image "path/to/image.jpg" \
    --question "What's happening in this image?"
```

**Test Summary Agent:**
```bash
python scripts/run_inference.py \
    --model "models/summary_agent/final" \
    --image "path/to/image.jpg" \
    --question "What's happening in this image?"
```

---

### Step 5: Launch Demo
```bash
python demo/app.py
```

**Access:**
- Local: `http://localhost:7860`
- Colab: Public Gradio link generated automatically

---

## 🚀 Google Colab: Complete Training Pipeline

Run everything in one session (~70 minutes):

### 1. Setup & Mount Drive
```python
from google.colab import drive
import os

drive.mount('/content/drive')
backup_dir = "/content/drive/MyDrive/insight-chain-models"
os.makedirs(backup_dir, exist_ok=True)
```

---

### 2. Clone Repository
```python
%cd /content
!rm -rf insight-chain
!git clone https://github.com/TejasCThakare/insight-chain.git
%cd insight-chain
!pip install --upgrade accelerate -q
```

---

### 3. Download & Prepare Data (~30 mins)
```bash
!python scripts/run_data_pipeline.py
```

**Verify data:**
```python
import json
with open('data/processed/train.json') as f:
    data = json.load(f)
print(f"✅ Data samples: {len(data)}")
print(f"📝 Sample: {data[0]['final_answer']}")
```

---

### 4. Train Reasoning Agent (~20 mins)
```bash
!python scripts/run_training.py --mode reasoning
```

**Backup to Drive:**
```python
if os.path.exists("models/reasoning_agent/final"):
    !du -sh models/reasoning_agent/final
    !cp -r models/reasoning_agent/final {backup_dir}/reasoning_agent_final
    print(f"✅ Backed up to: {backup_dir}/reasoning_agent_final")
```

---

### 5. Train Summary Agent (~20 mins)
```bash
!python scripts/run_training.py --mode summary
```

**Backup to Drive:**
```python
if os.path.exists("models/summary_agent/final"):
    !du -sh models/summary_agent/final
    !cp -r models/summary_agent/final {backup_dir}/summary_agent_final
    print(f"✅ Backed up to: {backup_dir}/summary_agent_final")
```

---

### 6. Test Both Agents
```python
test_image = "data/processed/images/aokvqa_125.jpg"
test_question = "What's happening in this image?"

print("🧠 REASONING AGENT:")
!python scripts/run_inference.py \
    --model "models/reasoning_agent/final" \
    --image {test_image} \
    --question "{test_question}"

print("\n🪶 SUMMARY AGENT:")
!python scripts/run_inference.py \
    --model "models/summary_agent/final" \
    --image {test_image} \
    --question "{test_question}"
```

---

### 7. Launch Demo
```bash
!python demo/app.py
```

**✅ Complete! Models backed up to:** `/content/drive/MyDrive/insight-chain-models/`

---

## 📈 Results

- **Inference Time**: ~60 seconds per query (T4 GPU)
- **Memory Usage**: 10GB peak (vs 16GB standard)
- **Training Time**: ~70 minutes total (Colab T4)

---

## 🧪 Python API
```python
from insight_chain import DualAgentVQA

model = DualAgentVQA(
    reasoning_adapter="models/reasoning_agent/final",
    summary_adapter="models/summary_agent/final"
)

result = model.predict(
    image="path/to/image.jpg",
    question="What do you see?"
)

print(result['reasoning'])  # Step-by-step analysis
print(result['summary'])    # Concise answer
```

---

## 🐛 Troubleshooting

**Problem: Data pipeline fails**
- Check internet connection and disk space (~10GB needed)
- Rerun: `!python scripts/run_data_pipeline.py`

**Problem: Training crashes (OOM)**
- Reduce batch size in training config
- Use Colab Pro for better GPU memory
- Enable gradient checkpointing

**Problem: `final_answer` field shows `[`**
- Data preparation failed - rerun Step 3
- Verify with the data validation code

---

## ✒️ Citation
```bibtex
@software{insight_chain_2025,
  author = {Thakare, Tejas},
  title = {Insight-Chain: Dual-Agent Visual Reasoning},
  year = {2025},
  url = {https://github.com/TejasCThakare/insight-chain}
}
```

---

## 🙏 Acknowledgments

- **Qwen2-VL** by Alibaba Cloud - Base vision-language model
- **LoRA** (Hu et al.) - Parameter-efficient fine-tuning
- **A-OKVQA** and **ScienceQA** - Training datasets
- **Hugging Face** - Transformers ecosystem

---

## 📧 Contact

**Tejas Thakare**  
- 🔗 GitHub: [@TejasCThakare](https://github.com/TejasCThakare)  
- 💼 LinkedIn: [Your Profile]  
- 📧 Email: your.email@example.com

🔍 **Currently seeking ML/Computer Vision opportunities!**

---

## 📜 License

MIT License © 2025 Tejas Thakare
