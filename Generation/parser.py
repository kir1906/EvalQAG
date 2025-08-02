import re
from collections import defaultdict

def clean_unicode_and_markdown(text):
    """
    Cleans unicode artifacts, formatting characters, and unnecessary markdown clutter.
    """
    text = re.sub(r'\\%', '__PERCENT__', text)
    text = re.sub(r'\\$', '__DOLLAR__', text)
    text = text.encode('utf-8', 'surrogatepass').decode('unicode_escape', errors='ignore')
    text = text.replace('__PERCENT__', '%').replace('__DOLLAR__', '$')
    text = re.sub(r'“|”', '"', text)
    text = re.sub(r'’|‘', "'", text)
    text = re.sub(r'–|—', '-', text)
    text = re.sub(r'\.{2,}', '.', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_pairs(text):
    """
    Extracts structured QA pairs from a formatted block of model-generated text.
    """
    def remove_whitespaces(temp): return re.sub(r"^[\s\-:]+", "", temp)

    res = []
    qString, aString, cString, csString = '**Question**', '**Answer**', '**Context**', '**Conditions**'

    # Normalize variants of QA labels
    for key in ['question', 'answer', 'context', 'conditions']:
        for variant in [f"{key}:", f"{key}-", f"{key} -", f"{key.upper()}:", f"{key.upper()}-", f"{key.upper()} -"]:
            text = text.replace(variant, key.capitalize())

    while text:
        pairs = defaultdict(str)
        qStart = text.find(qString)
        temp = text[qStart + len(qString):]
        temp = remove_whitespaces(temp)

        aStart = temp.find(aString)
        q = temp[:aStart]
        pairs['question'] = q.strip()
        temp = temp[aStart + len(aString):]
        temp = remove_whitespaces(temp)

        csStart = temp.find(csString)
        a = temp[:csStart]
        pairs['answer'] = a.strip()
        temp = temp[csStart + len(csString):]
        temp = remove_whitespaces(temp)

        cStart = temp.find(cString)
        conditions = temp[:cStart]
        pairs['conditions'] = conditions.strip()
        temp = temp[cStart + len(cString):]

        aStart = temp.find('###')
        c = temp[:aStart]
        pairs['context'] = c.strip()

        text = temp[aStart:]
        res.append(pairs)
        if aStart == -1: break

    return res
