import re

def split_large_chunk(text, max_chunk_size=4096, overlap=512):
    """
    Splits a large chunk of text into smaller overlapping chunks.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_chunk_size, len(text))
        chunks.append(text[start:end])
        start += (max_chunk_size - overlap)
    return chunks

def chunk_markdown(markdown_file, chunk_size=4096):
    """
    Reads a markdown file and splits it into manageable chunks based on headers and max size.
    """
    with open(markdown_file, encoding='utf-8') as f:
        markdown_text = f.read()

    markdown_text = re.sub(r"!\[.*?\]\(.*?\)", "", markdown_text)  # Remove images

    pattern = re.compile(r'(?=^(?:\s*)#\s)', re.MULTILINE)
    sections = pattern.split(markdown_text)
    sections = [section.strip() for section in sections if section.strip()]

    merged = []
    buffer = ""

    for section in sections:
        if len(buffer) + len(section) < chunk_size:
            buffer += "\n\n" + section
        else:
            if buffer:
                merged.append(buffer.strip())
            buffer = section

    if buffer:
        merged.append(buffer.strip())

    final_chunks = []
    for chunk in merged:
        if len(chunk) > 8192:
            split_chunks = split_large_chunk(chunk, max_chunk_size=4096, overlap=512)
            final_chunks.extend(split_chunks)
        else:
            final_chunks.append(chunk)

    return final_chunks
