import json
import pandas as pd

def load_squad_v2(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)

    rows = []
    for article in data['data']:
        for para in article['paragraphs']:
            context = para['context']
            for qa in para['qas']:
                if qa['is_impossible']:
                    continue
                question = qa['question']
                answer = qa['answers'][0]['text']
                rows.append({'context': context, 'question': question, 'answer': answer})
    return pd.DataFrame(rows)

if __name__ == "__main__":
    df = load_squad_v2("train-v2.0.json")
    print(df.head())

sampled_df = df.sample(n=100, random_state=42).reset_index(drop=True)

# Save only necessary columns
sampled_df[['question', 'context', 'answer']].to_csv('sampled_questions.csv', index=False)

print("sampled_questions.csv created with 100 question-context pairs.")
