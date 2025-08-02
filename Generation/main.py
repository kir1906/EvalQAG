import os
import sys
from tqdm import tqdm
from .config import models, question_types_list, fewshot_prompts
from .generator import extract_qa_pairs

def main(files_folder, output_folder, start_index):
    start_index = int(start_index)
    if not os.path.exists(files_folder):
        print(f'Error: {files_folder} does not exist!')
        return
    os.makedirs(output_folder, exist_ok=True)

    markdown_files = sorted(os.listdir(files_folder))[start_index:561]
    if not markdown_files:
        print(f'No files found in {files_folder}')
        return

    for file in tqdm(markdown_files, desc='Files'):
        markdown_path = os.path.join(files_folder, file)
        for model in tqdm(models, desc='Model', leave=False):
            for q in tqdm(question_types_list, desc=f"{file} | {model}", leave=False):
                extract_qa_pairs(markdown_path, output_folder, model, q, fewshot_prompts[(model, q)])

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print('Usage: python main.py <files_folder> <output_folder> <start_index>')
    else:
        main(sys.argv[1], sys.argv[2], sys.argv[3])
