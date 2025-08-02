import os
import json
from collections import defaultdict

# Define question types
question_types = ['Yes/No', 'Yes/No cond', 'Legal Obligation', 'Factual', 'Descriptive']
RELEVANCE_THRESHOLD = 7

def global_threshold_relevance(input_folder, output_path):
    type_to_triplets = defaultdict(list)
    print(len(os.listdir(input_folder)))
    
    # Walk through all files in all subfolders
    for root, _, files in os.walk(input_folder):
        for file in files:
            if not file.endswith('.json'):
                continue

            filepath = os.path.join(root, file)
            with open(filepath, 'r') as f:
                try:
                    blocks = json.load(f)
                except Exception as e:
                    print(f"Error loading {filepath}: {e}")
                    continue

            for block in blocks:
                qtype = block['question_type']
                doc_id = os.path.basename(root)
                chunk = block['chunk']

                for triplet in block['result']:
                    if ((triplet['relevance'] + triplet['intent'])/2) >= RELEVANCE_THRESHOLD and (triplet['accuracy']+triplet['completeness']+triplet['groundedness'])/3 >= RELEVANCE_THRESHOLD :
                        type_to_triplets[qtype].append({
                            "document_id": doc_id,
                            "chunk": chunk,
                            "question": triplet['question'],
                            "answer": triplet['answer'],
                            "context": triplet['context'],
                            "conditions": triplet['conditions'],
                            "relevance": triplet['relevance'],
                            "accuracy": triplet['accuracy'],
                            "completeness": triplet['completeness'],
                            "groundedness": triplet.get('groundedness', 0),
                            "intent_score": triplet['intent'],
                            "model": triplet['model']
                        })

    # Save filtered triplets by question type
    os.makedirs(output_path, exist_ok=True)
    for qtype in question_types:
        cleaned_qtype = qtype.replace('/', '-').replace(' ', '-')
        triplets = type_to_triplets[qtype]
        out_file = os.path.join(output_path, f'relevance5_{cleaned_qtype}.json')
        with open(out_file, 'w') as f:
            json.dump(triplets, f, indent=2)

        print(f"Wrote {len(triplets)} entries to {out_file}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Filter questions by relevance threshold per question type")
    parser.add_argument("input_folder", help="Path to folder with deduplicated outputs")
    parser.add_argument("output_folder", help="Path to folder for filtered output")

    args = parser.parse_args()
    global_threshold_relevance(args.input_folder, args.output_folder)
