import sys
import os
import json
import torch
from collections import defaultdict
from sentence_transformers import util
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import Settings
import pandas as pd
from tqdm import tqdm

ollama_embedding = OllamaEmbedding(model_name="llama3")
Settings.embed_model = ollama_embedding

# Model names (static list)
MODEL_NAMES = [
    "gemma3:27b",
    "llama3.3",
    "yi:34b",
    "mixtral:8x22b"
]

metric_types = ['accuracy', 'completeness',  'groundedness','relevance','intent']
question_types = ['Yes-No', 'Yes-No-cond', 'Legal-Obligation', 'Factual', 'Descriptive']
models = ['mixtral', 'gemma3:27b', 'llama3.3', 'yi:34b']

fewshot_prompts = {
    (models[0], question_types[0]): 3,
    (models[0], question_types[1]): 3,
    (models[0], question_types[2]): 5,
    (models[0], question_types[3]): 4,
    (models[0], question_types[4]): 2,
    (models[1], question_types[0]): 3,
    (models[1], question_types[1]): 5,
    (models[1], question_types[2]): 2,
    (models[1], question_types[3]): 3,
    (models[1], question_types[4]): 4,
    (models[2], question_types[0]): 4,
    (models[2], question_types[1]): 5,
    (models[2], question_types[2]): 3,
    (models[2], question_types[3]): 2,
    (models[2], question_types[4]): 2,
    (models[3], question_types[0]): 4,
    (models[3], question_types[1]): 5,
    (models[3], question_types[2]): 3,
    (models[3], question_types[3]): 4,
    (models[3], question_types[4]): 2,
}


def load_json_file(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)
    
def generate_embeddings(questions):
    return [ollama_embedding.get_text_embedding(q) for q in questions]


def deduplicate_utility(data, metadata):

    res = []
    
    for idx, triplets in data.items():

        questions = [item['question'] for item in triplets]
        
        embeddings = generate_embeddings(questions)
        clusters = defaultdict(list)

        n = len(questions)
        parent = list(range(n))

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            root_x = find(x)
            root_y = find(y)
            if root_x != root_y:
                parent[root_y] = root_x
        
        for i in range(n):
            for j in range(i+1, n):
                # try:
                sim = util.pytorch_cos_sim(
                torch.tensor(embeddings[i]).unsqueeze(0),
                torch.tensor(embeddings[j]).unsqueeze(0)
            ).item()


                if sim >= 0.95:
                    union(i, j)
        
        for i in range(n):
            clusters[find(i)].append(i)

        final_triplets = []

        for indices in clusters.values():
            best_question_entry = None
            best_question_intent = float('-inf')

            best_answer_entry = None
            best_answer_score = float('-inf')

            for index in indices:
                entry = triplets[index]

                # 1. Track best question by intent
                if entry['intent'] > best_question_intent:
                    best_question_intent = entry['intent']
                    best_question_entry = entry

                # 2. Track best answer by composite score
                answer_score = (
                    0.40 * entry['accuracy'] +
                    0.40 * entry['completeness'] +
                    0.20 * entry['groundedness']
                )
                if answer_score > best_answer_score:
                    best_answer_score = answer_score
                    best_answer_entry = entry

            if best_question_entry and best_answer_entry:
                final_triplets.append({
                    "question": best_question_entry["question"],
                    "answer": best_answer_entry["answer"],
                    "context": best_answer_entry["context"],
                    "conditions" : best_answer_entry["conditions"],
                    "accuracy": best_answer_entry["accuracy"],
                    "completeness": best_answer_entry["completeness"],
                    "groundedness": best_answer_entry["groundedness"],
                    "relevance": best_question_entry["relevance"],
                    "intent": best_question_entry["intent"],
                    "model": best_answer_entry["model"]
                })

        meta = metadata[idx]
        res.append({
            'block_index': idx,
            'chunk': meta['chunk'],
            'question_type': meta['question_type'],
            'document_id': meta['document_id'],
            'result': final_triplets

        })
    
    return res


def deduplicate_questions(files_folder, document_name, question_type, output_folder):
    qtype = question_type

    # Prepare output path
    document_output_folder = os.path.join(output_folder, document_name)
    os.makedirs(document_output_folder, exist_ok=True)

    output_file_path = os.path.join(document_output_folder, f'{document_name}_{qtype}.json')

    # Skip if file already exists
    if os.path.exists(output_file_path):
        return

    # chunk_key -> list of questions
    all_chunks_data = defaultdict(list)
    chunk_meta_info = {}

    for model in models:
        fname = f'{document_name}_{model}_{qtype}_{fewshot_prompts[(model, question_type)]}.json'
        fpath = os.path.join(files_folder, fname)

        if not os.path.exists(fpath):
            continue

        data = load_json_file(fpath)

        for block in data:
            chunk_key = block['chunk']
            triplets = []

            for resp in block['response']:
                if not resp['question'] or not resp['answer'] or not resp['context']:
                    continue

                triplets.append({
                    'question': resp['question'],
                    'answer': resp['answer'],
                    'context': resp['context'],
                    'conditions': resp['conditions'],
                    "accuracy": resp.get("accuracy_score", 0),
                    "completeness": resp.get("completeness_score", 0),
                    "groundedness": resp.get("groundedness_score", 0),
                    "relevance": resp.get("relevance_score", 0),
                    "intent": resp.get("intent_score", 0),
                    "model": model
                })

            all_chunks_data[chunk_key].extend(triplets)

            if chunk_key not in chunk_meta_info:
                chunk_meta_info[chunk_key] = {
                    'chunk': block['chunk'],
                    'question_type': block['question_type'],
                    'document_id': block['document_id']
                }

    # convert chunk_key -> index for consistent output
    chunk_to_index = {chunk: idx for idx, chunk in enumerate(sorted(all_chunks_data))}
    indexed_data = {chunk_to_index[chunk]: triplets for chunk, triplets in all_chunks_data.items()}
    indexed_meta = {chunk_to_index[chunk]: meta for chunk, meta in chunk_meta_info.items()}

    res = deduplicate_utility(indexed_data, indexed_meta)

    with open(output_file_path, 'w') as f:
        json.dump(res, f, indent=4)




def main(qa_folder, out_folder, start_index):
    files = set()

    # Each subfolder is a document folder (name = document ID)
    print(len(os.listdir(qa_folder)))
    for document_dir in os.listdir(qa_folder):
        document_path = os.path.join(qa_folder, document_dir)

        if not os.path.isdir(document_path):
       
            continue  # Skip if it's not a folder

        for file in os.listdir(document_path):
            if not file.endswith('.json'):
             
                continue

            parts = file.rsplit('.', 1)[0].split('_')
            if len(parts) < 4:
               
                continue  # Skip malformed filenames

            question_type = parts[-2]
           
            files.add((document_dir, question_type))  # document_dir is document_name
    
    files = sorted(list(files))
    
    for document_name, question_type in tqdm(files[start_index:], desc='Files'):
        deduplicate_questions(
            os.path.join(qa_folder, document_name),  # Path to the folder containing files
            document_name,
            question_type,
            out_folder
        )


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print('Usage: python main.py <qa_folder> <output_folder> <start_index>')
    else:
        main(sys.argv[1], sys.argv[2], int(sys.argv[3]))
