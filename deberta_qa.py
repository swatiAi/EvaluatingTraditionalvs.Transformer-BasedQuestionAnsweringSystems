from transformers import pipeline

def load_deberta_model():
    qa_pipeline = pipeline(
        "question-answering",
        model="deepset/deberta-v3-base-squad2",
        tokenizer="deepset/deberta-v3-base-squad2"
    )
    return qa_pipeline

def deberta_qa(qa_pipeline, context, question):
    result = qa_pipeline(question=question, context=context)
    return result['answer'], result['score']




