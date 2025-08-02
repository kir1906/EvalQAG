import pandas as pd

# Data files
FINAL_DF_PATH = './data/final_df.json'
CONTEXT_CSV_PATH = './data/context.csv'

# Load metadata
final_df = pd.read_json(FINAL_DF_PATH)
context_df = pd.read_csv(CONTEXT_CSV_PATH)

# Unique categories
policy_types = context_df['policy_type'].unique()
question_types = context_df['question_type'].unique()

# Constants
CHUNK_SIZE = 4096
CHUNK_OVERLAP = 512
MAX_CHUNK_LIMIT = 8192
TOKEN_PER_QUESTION = 1024

# Models and question types
models = ['mixtral', 'gemma3:27b', 'llama3.3', 'yi:34b']
question_types_list = ['Yes/No', 'Yes/No cond', 'Legal Obligation', 'Factual', 'Descriptive']

# Few-shot prompt config
fewshot_prompts = {
    (models[0], question_types_list[0]): 3,
    (models[0], question_types_list[1]): 3,
    (models[0], question_types_list[2]): 5,
    (models[0], question_types_list[3]): 4,
    (models[0], question_types_list[4]): 2,
    (models[1], question_types_list[0]): 3,
    (models[1], question_types_list[1]): 5,
    (models[1], question_types_list[2]): 2,
    (models[1], question_types_list[3]): 3,
    (models[1], question_types_list[4]): 4,
    (models[2], question_types_list[0]): 4,
    (models[2], question_types_list[1]): 5,
    (models[2], question_types_list[2]): 3,
    (models[2], question_types_list[3]): 2,
    (models[2], question_types_list[4]): 2,
    (models[3], question_types_list[0]): 4,
    (models[3], question_types_list[1]): 5,
    (models[3], question_types_list[2]): 3,
    (models[3], question_types_list[3]): 4,
    (models[3], question_types_list[4]): 2,
}

# Document exclusions
EXCLUDED_FILES = ['']
