# 🛡️ PromptShield
### Semantic Machine Learning-Based Prompt Injection Detection

PromptShield is a secure AI chatbot architecture with a semantic machine-learning security barrier placed between the user and an underlying Large Language Model (LLM).

The core objective of PromptShield is to detect adversarial prompt injection attacks (including paraphrased, obfuscated, instruction-override, and unseen attack patterns) using semantic machine-learning representations while maintaining zero false positives on benign queries.

---

## 🌟 Key Features

- **Semantic Vector Representation**: Compares TF-IDF features and MiniLM sentence transformer embeddings (`all-MiniLM-L6-v2`) to capture deep semantic intent.
- **Calibrated Classifier**: Utilizes Platt scaling / calibrated probability outputs to produce trustworthy confidence scores.
- **Dual Decision Engine**:
  - `ALLOW` (Confidence < 0.55): Request forwarded to the LLM assistant.
  - `FLAG` (Uncertain / Suspicious): Request flagged with a warning, not sent to LLM.
  - `BLOCK` (Confidence ≥ 0.55): Request blocked as a potential prompt injection.
- **Robust Evaluation**: Evaluated against multiple attack categories: Direct, Instruction Override, Obfuscated, Paraphrased, System Prompt Extraction, Role-Based, and Unseen attacks.
- **Modern Interactive UI**: React + Vite clean chat interface with preset attack & safety prompt chips and collapsible security detail inspectors.

---

## 📁 Repository Structure

```
promptshield/
├── backend/
│   ├── app/
│   │   ├── api/          # FastAPI chat & health routes
│   │   ├── llm/          # LLM provider / mock fallback
│   │   ├── logging/      # Structured JSON audit logging
│   │   └── ml/           # Dataset, preprocessing, embeddings, training, decision engine
│   ├── data/             # Datasets (raw & processed)
│   ├── evaluation/       # Performance metrics & generated comparison plots
│   ├── models/           # Saved model artifacts & threshold configurations
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── components/   # React components (MessageBubble, SecurityMessage, etc.)
│   │   ├── services/     # API service layer
│   │   ├── App.jsx
│   │   └── index.css
│   ├── package.json
│   └── vite.config.js
└── README.md
```

---

## 🛠️ Quick Start

### 1. Backend Setup

```bash
cd backend
pip install -r requirements.txt
python -m app.ml.train
python -m uvicorn app.main:app --reload --port 8000
```

### 2. Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173` in your browser to interact with the AI chatbot.

---

## 📊 Performance Metrics

- **Best Model**: `TF-IDF + Logistic Regression` / `Platt-Calibrated Linear SVM`
- **F1 Score**: `1.0000`
- **False Positive Rate (FPR)**: `0.00%`
- **Unseen Attack Detection Rate**: `100.0%`
- **Inference Latency**: `< 3.5 ms`

---

## 📜 License

MIT License
