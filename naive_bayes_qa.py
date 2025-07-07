import nltk
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

nltk.download('punkt')
nltk.download('punkt_tab')

def prepare_nb_data(df, limit=1000):
    X = []
    y = []
    for i, row in df.iterrows():
        context = row['context']
        question = row['question']
        true_answer = row['answer'].strip().lower()

        for sent in sent_tokenize(context):
            sent_clean = sent.strip().lower()
            combined_input = question + " " + sent
            label = 1 if true_answer in sent_clean else 0
            X.append(combined_input)
            y.append(label)

        if len(X) > limit:
            break
    return X, y

def train_nb_model(X_train, y_train):
    nb_clf = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('nb', MultinomialNB())
    ])
    nb_clf.fit(X_train, y_train)
    return nb_clf

def naive_bayes_qa(nb_clf, context, question):
    sentences = sent_tokenize(context)
    combined_inputs = [question + " " + sent for sent in sentences]
    probs = nb_clf.predict_proba(combined_inputs)[:, 1]
    best_idx = probs.argmax()
    return sentences[best_idx], probs[best_idx]
