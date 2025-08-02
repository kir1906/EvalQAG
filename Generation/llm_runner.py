from llama_index.llms.ollama import Ollama

def sanitize_filename(name):
    """
    Replaces illegal filename characters with underscores.
    """
    import re
    return re.sub(r'[:<>"/\\|?*]', '_', name)

def init_llm(model_name):
    """
    Initializes an Ollama LLM with a given model name.
    """
    return Ollama(model=model_name, request_timeout=12000, verbose=True)
