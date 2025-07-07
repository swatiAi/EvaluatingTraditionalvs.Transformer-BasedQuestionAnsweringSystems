
# 🧠 Evaluating Traditional vs. Transformer-Based Question Answering Systems

This project compares two distinct approaches to building Question Answering (QA) systems:

- ✅ **Traditional Method**: TF-IDF + Naive Bayes
- 🤖 **Modern Method**: DeBERTa-v3 (Transformer-based model)

Using the SQuAD v2.0 dataset, both models are evaluated on the basis of **accuracy**, **confidence**, and **inference time**, and presented in a user-friendly **Streamlit web interface**.

---

## 📌 Table of Contents

- [📘 Project Overview](#-project-overview)
- [📊 Technologies Used](#-technologies-used)
- [🧠 Models and Logic](#-models-and-logic)
- [📁 File Structure](#-file-structure)
- [🚀 How to Run Locally](#-how-to-run-locally)
- [📸 Screenshots](#-screenshots)
- [📈 Results](#-results)
- [🔍 Future Work](#-future-work)

---

## 📘 Project Overview

This project is part of our NLP (AI-401) Final Year Research. It explores the effectiveness of:
- A lightweight, rule-based model for short QA
- A transformer-based model for deep contextual understanding

💡 The models are deployed in a **Streamlit app** to compare their outputs side-by-side with live metrics.

---

## 📊 Technologies Used

| Tool | Purpose |
|------|---------|
| `Python` | Core programming language |
| `Scikit-learn` | Naive Bayes implementation |
| `Transformers` (HuggingFace) | DeBERTa-v3 model |
| `NLTK` | Text preprocessing |
| `Pandas`, `NumPy` | Data manipulation |
| `Streamlit` | Web-based UI |
| `Matplotlib` | Optional result graphing |
| `SQuAD v2.0` | Benchmark QA dataset |

---

## 🧠 Models and Logic

### 🔹 Naive Bayes QA
- Uses TF-IDF vectorization over sentence-tokenized context
- Predicts the most likely sentence containing the answer

### 🔹 DeBERTa v3 QA
- Pre-trained transformer fine-tuned on SQuAD v2
- Uses `pipeline("question-answering")` from Hugging Face

---

## 📁 File Structure

```bash
.
├── app.py                 # Main Streamlit app (single model)
├── app_compare.py         # Streamlit app for NB vs DeBERTa comparison
├── preprocess.py          # Loads and samples SQuAD v2 data
├── naive_bayes_qa.py      # NB model logic
├── deberta_qa.py          # DeBERTa pipeline wrapper
├── sampled_questions.csv  # Processed subset of SQuAD v2
├── requirements.txt       # Required Python libraries
└── README.md              # This file
```

---

## 🚀 How to Run Locally

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/qa-comparison-project.git
cd qa-comparison-project
```

### 2. Set up Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\activate on Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit App

```bash
streamlit run app_compare.py
```

---

## 📸 Screenshots

> 📌 Add these after running the app:
- Streamlit UI interface
- Side-by-side answers from both models
- Console showing inference time
- Comparison metrics or results summary

---

## 📈 Results (Sample)

| Metric             | Naive Bayes     | DeBERTa v3         |
|--------------------|------------------|---------------------|
| Accuracy           | ~62%             | ~87%                |
| Inference Time     | 0.03 sec         | 1.4 sec             |
| Confidence Score   | Lower            | Higher              |
| Model Size         | <1MB             | ~450MB              |

---

## 🔍 Future Work

- Add semantic answer evaluation (e.g., BLEU/F1 score)
- Use passage retrieval for open-domain QA
- Try lightweight transformers (DistilBERT, MiniLM)
- Deploy app using Streamlit Cloud / Hugging Face Spaces

---

## 📬 Contact

- **Usman Swati**  
- Final Year BSCS, Spring 2025  
- GitHub: [@yourusername](https://github.com/yourusername)
