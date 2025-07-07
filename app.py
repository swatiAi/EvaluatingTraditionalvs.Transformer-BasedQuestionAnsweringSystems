import streamlit as st
import pandas as pd
import time
import os
import nltk
from naive_bayes_qa import prepare_nb_data, train_nb_model, naive_bayes_qa
from deberta_qa import load_deberta_model, deberta_qa

nltk.download('punkt')

# Load sampled questions CSV
df = pd.read_csv("sampled_questions.csv")

st.title("QA System Comparison: Naive Bayes vs DeBERTa")

# Select question from dropdown
question = st.selectbox("Select a question:", df['question'].tolist())
context = df[df['question'] == question]['context'].values[0]
true_answer = df[df['question'] == question]['answer'].values[0]

st.write("### Context")
st.write(context)

# Cache and load Naive Bayes model
@st.cache_resource
def load_nb_and_train(data):
    X_train, y_train = prepare_nb_data(data, limit=5000)
    clf = train_nb_model(X_train, y_train)
    return clf

nb_clf = load_nb_and_train(df)

# Cache and load DeBERTa model
@st.cache_resource
def load_deberta():
    return load_deberta_model()

deberta_pipeline = load_deberta()

def measure_inference_time_nb(nb_clf, context, question, repeats=10):
    start = time.time()
    for _ in range(repeats):
        _ = naive_bayes_qa(nb_clf, context, question)
    end = time.time()
    return (end - start) / repeats

def measure_inference_time_deberta(qa_pipeline, context, question, repeats=10):
    start = time.time()
    for _ in range(repeats):
        _ = qa_pipeline(question=question, context=context)
    end = time.time()
    return (end - start) / repeats

def get_model_size_mb(path):
    total_size = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size / (1024 * 1024)  # MB

if st.button("Get Answers"):

    # Predict with Naive Bayes and time it
    nb_time = measure_inference_time_nb(nb_clf, context, question)
    nb_answer, nb_conf = naive_bayes_qa(nb_clf, context, question)

    # Predict with DeBERTa and time it
    deberta_time = measure_inference_time_deberta(deberta_pipeline, context, question)
    deberta_answer, deberta_conf = deberta_qa(deberta_pipeline, context, question)

    st.subheader("Naive Bayes")
    st.write(f"Answer: {nb_answer}")
    st.write(f"Confidence: {nb_conf:.4f}")
    st.write(f"Inference time: {nb_time:.4f} seconds")

    st.subheader("DeBERTa v3")
    st.write(f"Answer: {deberta_answer}")
    st.write(f"Confidence: {deberta_conf:.4f}")
    st.write(f"Inference time: {deberta_time:.4f} seconds")

    exact_match = (nb_answer.strip().lower() == deberta_answer.strip().lower())
    st.subheader("Comparison")
    st.write(f"Do both models give the same answer? {'✅ Yes' if exact_match else '❌ No'}")
    st.write(f"True Answer: {true_answer}")

    # Approximate Naive Bayes model size
    nb_model_size = 0.01  # MB (very small)

    deberta_cache_path = os.path.expanduser("~/.cache/huggingface/hub/models--deepset--deberta-v3-base-squad2")
    deberta_model_size = get_model_size_mb(deberta_cache_path)

    st.subheader("Model Sizes")
    st.write(f"Naive Bayes model size (approx.): {nb_model_size} MB")
    st.write(f"DeBERTa model size: {deberta_model_size:.2f} MB")
    
    
