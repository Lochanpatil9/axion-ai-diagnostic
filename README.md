# 🚗 AXION EV AI Diagnostic System

> A complete end-to-end AI engineering project built over 6 weeks —
> covering ML, Deep Learning, NLP, LLMs, RAG, and MLOps —
> applied to real-time fault diagnosis for the AXION autonomous electric vehicle.

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green)
![Docker](https://img.shields.io/badge/Docker-ready-blue)
![MLflow](https://img.shields.io/badge/MLflow-tracked-purple)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📌 What is this?

AXION is a real autonomous electric vehicle project built on a Maruti 800
platform using ROS2 Jazzy, Raspberry Pi 5, and YDLIDAR X4 Pro at MANIT Bhopal.

This repository contains an AI diagnostic system built on top of that hardware —
classifying motor and battery faults in real time using a multi-agent architecture
that combines classical ML, deep learning, NLP, and LLM-based retrieval.

Everything here was built from scratch over 6 weeks of implementation-first learning.
Every component is understood line by line — not copied from tutorials.

---

## 🏗️ System Architecture

```
Raw Sensor Data
(battery_voltage, phase_current, motor_temp, vehicle_speed, rpm, soc)
        │
        ▼
┌───────────────────────────────────────────────────────┐
│                 AXION AI Orchestrator                 │
│                                                       │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────┐  │
│  │   Fault     │  │   Urgency    │  │     RAG     │  │
│  │ Classifier  │  │  Detector    │  │  Knowledge  │  │
│  │ (XGBoost)   │  │   (BERT)     │  │  (FAISS)    │  │
│  └─────────────┘  └──────────────┘  └─────────────┘  │
│                                                       │
│  ┌───────────────────────────────────────────────┐    │
│  │              Math Agent                       │    │
│  │   power · SOC status · temperature analysis   │    │
│  └───────────────────────────────────────────────┘    │
└───────────────────────────────────────────────────────┘
        │
        ▼
Unified Diagnosis + Priority Level + Recommended Actions
        │
        ▼
  FastAPI REST API ──► Docker Container ──► Production
```

---

## 📁 Repository Structure

```
axion-ai-diagnostic/
│
├── week1_ml/
│   ├── titanic_pipeline.ipynb            # sklearn pipeline fundamentals
│   ├── credit_risk_classifier.ipynb      # full ML pipeline with evaluation
│   └── axion_fault_classifier.ipynb      # EV sensor fault classification
│
├── week2_deep_learning/
│   ├── neural_net_scratch.ipynb          # PyTorch neural net from scratch
│   ├── mnist_classifier.ipynb            # image classification + DataLoader
│   ├── cnn_mnist.ipynb                   # CNN architecture + feature maps
│   └── dl_pipeline_template.py           # reusable training template
│
├── week3_nlp/
│   ├── bow_tfidf.ipynb                   # Bag of Words + TF-IDF
│   ├── word_embeddings.ipynb             # embeddings + PCA visualization
│   ├── attention_mechanism.ipynb         # attention from scratch
│   ├── bert_finetuning.ipynb             # HuggingFace fine-tuning on IMDb
│   └── nlp_pipeline.py                   # reusable NLP inference pipeline
│
├── week4_llm_rag/
│   ├── llm_fundamentals.ipynb            # tokens, temperature, generation
│   ├── prompt_engineering.ipynb          # zero-shot, few-shot, CoT
│   ├── semantic_search.ipynb             # sentence-transformers + cosine sim
│   ├── rag_from_scratch.ipynb            # full RAG pipeline manual build
│   └── rag_pipeline.py                   # reusable RAGPipeline class
│
├── week5_mlops/
│   ├── app.py                            # FastAPI application
│   ├── Dockerfile                        # Docker container definition
│   ├── requirements.txt                  # all dependencies
│   ├── mlflow_tracking.ipynb             # experiment tracking + registry
│   ├── drift_monitoring.ipynb            # KS test data drift detection
│   └── mlops_pipeline.py                 # reusable MLOpsPipeline class
│
├── week6_agents/
│   ├── multi_agent_system.ipynb          # 4 agents + orchestrator
│   ├── lora_finetuning.ipynb             # LoRA fine-tuning on EV fault text
│   ├── llm_evaluation.ipynb              # evaluation framework
│   └── axion_ai_system.py               # complete capstone system
│
├── axion_model/
│   ├── fault_classifier.pkl              # trained XGBoost pipeline
│   ├── label_encoder.pkl                 # label encoder
│   └── feature_names.pkl                 # feature list
│
├── axion_rag/
│   ├── faiss_index.bin                   # FAISS vector index
│   └── chunks.pkl                        # document chunks
│
└── README.md
```

---

## 🧠 Week by Week Breakdown

### ⚙️ Week 1 — Machine Learning Pipelines

**Core concepts:** sklearn Pipeline, ColumnTransformer, feature engineering,
cross-validation, hyperparameter tuning, class imbalance

**What was built:**
- Full sklearn pipeline with separate preprocessing for numeric and categorical features
- Domain-specific feature engineering for EV sensor data:
  - `power_w = battery_voltage × phase_current`
  - `thermal_stress = motor_temp × phase_current / 100`
  - `voltage_efficiency = battery_voltage / 72.0`
- XGBoost fault classifier across 5 fault types
- SMOTE oversampling for imbalanced fault data
- GridSearchCV hyperparameter tuning

**Key result:** 96%+ weighted F1 on EV fault classification

---

### 🧬 Week 2 — Deep Learning with PyTorch

**Core concepts:** neural networks, backpropagation, training loop,
CNNs, feature maps, DataLoader, GPU training

**What was built:**
- Neural network from scratch — forward pass, loss, backprop, optimizer
- Complete training loop: `zero_grad → backward → step`
- MNIST classifier with DataLoader and batching
- CNN with `Conv2d`, `MaxPool2d` — spatial feature extraction
- Feature importance via gradient-based and permutation methods
- Reusable `FlexibleNN` template configurable from a dict

**Key result:** CNN achieving 99%+ accuracy on MNIST in 10 epochs

---

### 💬 Week 3 — Natural Language Processing

**Core concepts:** tokenization, TF-IDF, word embeddings,
attention mechanism, transformer fine-tuning

**What was built:**
- Bag of Words and TF-IDF sentiment classifier
- Word embeddings from scratch with PCA visualization
- Single-head attention from scratch with Q, K, V projections
- Attention weight visualization — which words the model focuses on
- DistilBERT fine-tuned on real IMDb data using plain PyTorch (no Trainer)
- Reusable `NLPInferencePipeline` class with save/load

**Key result:** Fine-tuned sentiment classifier — 92%+ F1 on IMDb

---

### 🤖 Week 4 — LLMs and RAG Systems

**Core concepts:** autoregressive generation, temperature,
prompt engineering, semantic search, FAISS, retrieval augmented generation

**What was built:**
- GPT-2 text generation with temperature experiments
- Prompt engineering patterns: zero-shot, few-shot, chain of thought
- Semantic similarity search with sentence-transformers
- FAISS vector database — build index, add vectors, similarity search
- Complete RAG pipeline from scratch:
  - Document chunking with overlap
  - Embedding all chunks
  - Query → retrieve → generate
- Reusable `RAGPipeline` class with save/load

**Key result:** AXION document Q&A system grounded in EV technical docs

---

### 🚀 Week 5 — MLOps and Production

**Core concepts:** REST APIs, containerization,
experiment tracking, data drift, model monitoring

**What was built:**
- FastAPI with Pydantic input validation
- Three endpoints: `/health`, `/predict`, `/predict/batch`
- Docker container — full app + model + dependencies packaged
- MLflow experiment tracking across three model variants
- Model registry — version, compare, promote best model
- KS test drift detection comparing training vs production distributions
- Prediction drift monitoring — catches distribution shift over time
- Reusable `MLOpsPipeline` class

**Key result:** Complete production pipeline —
train → track → serve → monitor → retrain

---

### 🧩 Week 6 — Multi-Agent Systems and Advanced GenAI

**Core concepts:** agent orchestration, LoRA fine-tuning,
LLM evaluation metrics, system integration

**What was built:**

**Four specialized agents:**
- `FaultClassifierAgent` — XGBoost pipeline from Week 1
- `UrgencyDetectorAgent` — BERT NLP from Week 3
- `RAGKnowledgeAgent` — FAISS retrieval from Week 4
- `MathAgent` — domain calculations (power, SOC, temp status)

**Orchestrator** — coordinates all agents, synthesizes unified diagnosis

**LoRA fine-tuning** — fine-tune DistilBERT on EV fault descriptions
with less than 1% of total parameters updated

**LLM evaluation framework:**
- Semantic similarity — captures paraphrasing via embeddings
- Faithfulness — checks answer is grounded in retrieved context
- Relevance — checks answer addresses the actual question
- Keyword coverage — verifies domain terms appear in answer

**`AXIONAISystem`** — all 6 weeks active in one class

**Key result:** Complete multi-agent AI diagnostic system
with session-level quality evaluation

---

## 🚀 Quick Start

### Option 1 — Run the API locally

```bash
# Clone
git clone https://github.com/Lochsnpstil9/axion-ai-diagnostic.git
cd axion-ai-diagnostic

# Install
pip install -r requirements.txt

# Start
uvicorn week5_mlops.app:app --host 0.0.0.0 --port 8000
```

### Option 2 — Run with Docker

```bash
docker build -t axion-api .
docker run -p 8000:8000 axion-api
```

### Option 3 — Run the full AI system

```python
from week6_agents.axion_ai_system import AXIONAISystem

system = AXIONAISystem()

result = system.diagnose(
    sensor_data={
        'battery_voltage': 56.0,
        'phase_current':   18.0,
        'motor_temp':      50.0,
        'vehicle_speed':   18.0,
        'rpm':             950.0,
        'soc':             12.0
    },
    user_text="Vehicle very slow battery seems almost dead"
)
```

---

## 🔌 API Reference

### `POST /predict`

**Request body:**
```json
{
  "battery_voltage": 56.0,
  "phase_current":   18.0,
  "motor_temp":      50.0,
  "vehicle_speed":   18.0,
  "rpm":             950.0,
  "soc":             12.0
}
```

**Response:**
```json
{
  "fault":       "VOLTAGE_DROP",
  "confidence":  0.9423,
  "alert":       true,
  "all_classes": {
    "NORMAL":       0.021,
    "OVERHEAT":     0.014,
    "OVERCURRENT":  0.009,
    "VOLTAGE_DROP": 0.942,
    "SENSOR_FAULT": 0.014
  }
}
```

### `POST /predict/batch`

Send a list of readings, get a list of predictions back.

### `GET /health`

```json
{
  "status": "healthy",
  "model":  "XGBoost fault classifier"
}
```

---

## 🤖 Multi-Agent Diagnosis Example

```
Input:       70V · 25A · 88°C · SOC 40%
Operator:    "Smoke from motor area, stopped immediately"

╔══════════════════════════════════════════╗
║       AXION DIAGNOSTIC REPORT           ║
╠══════════════════════════════════════════╣
║  🚨 Priority:    CRITICAL               ║
║  🔧 Fault:       OVERHEAT (94%)         ║
║  ⚡ Power:       1.75 kW                ║
║  🔋 Battery:     NORMAL  (40% SOC)      ║
║  🌡️  Temperature: CRITICAL (88°C)        ║
║  📢 Urgency:     CRITICAL               ║
╠══════════════════════════════════════════╣
║  Recommended actions:                   ║
║  • Check cooling system, reduce load    ║
║  • Motor temp above 80°C triggers       ║
║    thermal protection cutoff            ║
╚══════════════════════════════════════════╝
```

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Machine Learning | scikit-learn, XGBoost, imbalanced-learn |
| Deep Learning | PyTorch, torchvision |
| NLP | HuggingFace Transformers, PEFT |
| LLMs + RAG | sentence-transformers, FAISS, GPT-2 |
| API | FastAPI, Uvicorn, Pydantic |
| MLOps | MLflow, Docker, DVC |
| Monitoring | SciPy KS test, custom drift detection |
| Hardware | Raspberry Pi 5, YDLIDAR X4 Pro, ROS2 Jazzy |
| Language | Python 3.11+ |

---

## 📊 Results

| Component | Metric | Result |
|---|---|---|
| Fault Classifier (XGBoost) | Weighted F1 | 96%+ |
| CNN Image Classifier | Test Accuracy | 99%+ |
| BERT Sentiment Analysis | F1 Score | 92%+ |
| RAG Faithfulness Score | Grounding | 0.85+ |
| API Response Time | Latency | <100ms |
| LoRA Fine-tuning | Trainable Params | <1% of total |
| Drift Detection | Method | KS test p<0.05 |

---

## 🔋 Fault Types Detected

| Fault | Description | Key Signals |
|---|---|---|
| `NORMAL` | All systems operating correctly | Voltage 65–80V, temp 40–65°C |
| `OVERHEAT` | Motor winding temperature too high | Temp >80°C, elevated current |
| `OVERCURRENT` | Phase current spike | Current >40A, voltage sag |
| `VOLTAGE_DROP` | Battery voltage sag | Voltage <60V, low SOC |
| `SENSOR_FAULT` | Erratic sensor readings | All readings outside normal range |

---

## ⚙️ Engineered Features

Beyond raw sensor readings, the ML model uses domain-specific features:

```python
power_w            = battery_voltage × phase_current
thermal_stress     = motor_temp × phase_current / 100
voltage_efficiency = battery_voltage / 72.0
speed_rpm_ratio    = vehicle_speed / rpm
current_per_soc    = phase_current / soc
overheat_risk      = (temp > 70) × 2 + (current > 30)
```

These encode electrical engineering domain knowledge directly
into the feature space — improving F1 by 12% over raw features alone.

---

## 🎓 Learning Context

This project was built as part of a structured 6-week AI engineering
curriculum — implementation-first, no skipping steps.

The curriculum was designed to go from:
> *"I know basic ML but struggle to implement it"*

to:
> *"I can build and deploy a complete production AI system"*

Each week built exactly one layer of the final system.
Nothing was throwaway. Every component from Week 1 through Week 5
is active inside the Week 6 `AXIONAISystem`.

---

## 🔮 What's Next

- [ ] Connect real AXION ROS2 sensor topics to the FastAPI endpoint
- [ ] Replace GPT-2 with a quantized Llama model for better generation
- [ ] Add real-time dashboard with Streamlit or Grafana
- [ ] Deploy to cloud (Railway / Render) with permanent public URL
- [ ] Collect real AXION driving data and retrain on actual sensor logs
- [ ] Add time-series windowing for sequential fault detection

---


**Core interests:** Autonomous systems · Power electronics ·
AI/ML engineering · Data analytics

**Stack:** Python · PyTorch · scikit-learn · HuggingFace ·
FastAPI · Docker · MLflow · ROS2 · MATLAB/Simulink · C/C++

---

## 📄 License

MIT License — free to use, modify, and distribute with attribution.

---

<div align="center">

**Built with 6 weeks of implementation-first learning.**
**Every line of code understood, not just copied.**

*If this helped you — give it a ⭐*

</div>
