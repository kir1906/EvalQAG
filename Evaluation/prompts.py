import json
from .config import metric_types, question_types

# All your get_*_conditions functions
def get_accuracy_conditions(question_type):
    if question_type == question_types[0]:  # Yes-No
        return '''
- Ensure the answer clearly matches the information in the chunk. The answer must be either "Yes" or "No".
- Penalise the answer that provides additional information other than clear Yes or No.
- Penalize any answer that provides reasoning inconsistent with the chunk.
- If the chunk is ambiguous or silent, the answer should reflect uncertainty, not make definitive claims.
'''
    elif question_type == question_types[1]:  # Yes-No w Condition
        return '''
- The answer must be either "Yes" or "No" and must logically follow from the chunk, **assuming the provided condition holds**.
- Also evaluate the condition itself: Is it a meaningful and valid conditional clause (not a statement or fact)?,
Does it provide new context beyond what's already in the question?,
Is it supported, inferable, or verifiable from the chunk?
- Penalize if: If conditions empty(None) given in the input.
- Penalize if: The answer contradicts the chunk under the condition.
- Penalize if: Answer is other than "Yes" or "NO" 
- Penalize if: The condition is redundant, vague, or not truly conditional or empty(None).
- Penalize if: The chunk does not support drawing a conclusion under the stated condition.
'''
    elif question_type == question_types[2]:  # Legal-Obligation
        return '''
- Verify that the answer accurately reflects mandatory vs optional obligations.
- Terms like "must", "shall", "is required to" indicate obligation, while "may", "can", or "is encouraged to" do not.
- Deduct points if the answer exaggerates or softens the level of obligation stated in the chunk.
'''
    elif question_type == question_types[3]:  # Factual
        return '''
- Check that all stated facts (names, numbers, dates, procedures, etc.) match the chunk exactly.
- No guessing or inferring beyond what is explicitly mentioned.
- Omission or hallucination of any detail must be penalized.
'''
    return '''
- Confirm the answer captures the full scope and nuance of the description in the chunk.
- It should summarize, paraphrase, or explain the original information without adding new content or misinterpreting.
- Penalize vague, incomplete, or overly broad answers that dilute the document's intent.
'''

def get_completeness_conditions(question_type):
    if question_type == question_types[0]:  # Yes-No
        return '''
- If the question is unconditional, the answer must clearly respond with "Yes" or "No" and reflect reasoning supported by the document.
- Answers missing relevant justifications or stating unsupported claims in answer are incomplete.
'''

    elif question_type == question_types[1]:  # Yes-No w Condition
        return '''
- The answer must be a clear "Yes" or "No", but completeness also depends on whether this response covers the outcome **given the provided condition**.
- The chunk must support the answer under the stated condition.
- Penalize if the chunk includes important exceptions or qualifications that are not accounted(in conditions) for in the "Yes" or "No" choice.
'''


    elif question_type == question_types[2]:  # Legal-Obligation
        return '''
- The answer must include who is obligated, what is required, and under what conditions.
- It should distinguish mandatory actions from suggestions or permissions.
- Incompleteness arises if any element of the obligation is omitted or summarized too vaguely.
'''

    elif question_type == question_types[3]:  # Factual
        return '''
- All specific pieces of information asked (e.g., values, names, rules, timelines) must be covered.
- If the question implies a list or set of facts, the answer must include all of them.
- Omission of any key fact or partial listing reduces the completeness score.
'''

    return '''
- The answer must provide a comprehensive explanation or summary grounded in the document.
- All relevant aspects must be covered, not just a narrow or superficial part.
- Missing key details, interpretations, or oversimplifications indicate incompleteness.
'''

def get_intent_conditions(question_type):
    if question_type == question_types[0]:  # Yes-No
        return '''
- The question should be clearly answerable with a "Yes" or "No" based on the chunk.
- It must directly reflect the document's core assertion without adding ambiguity.
- Avoid vague or overly broad formulations; precision is essential.
- Penalize questions that are grammatically incorrect or poorly phrased.
'''

    elif question_type == question_types[1]:  # Yes-No w Condition
        return '''
- The question should not include any conditions or statements if a condition is not required to answer it.
- Penalize grammatically awkward or logically inconsistent questions.
'''

    elif question_type == question_types[2]:  # Legal-Obligation
        return '''
- The question should explicitly ask about legal obligations, duties, or compliance as described in the chunk.
- Check if the question uses precise legal terminology and reflects the scope of obligations accurately.
- Avoid vague or generalized legal language that is not supported by the document.
- Penalize poorly structured legal questions or those that conflate obligations with permissions.
'''

    elif question_type == question_types[3]:  # Factual
        return '''
- The question should request a specific fact, such as a date, value, term, or entity clearly stated in the chunk.
- It should be direct, unambiguous, and precise in its formulation.
- Penalize questions that are too broad, unclear, or could have multiple interpretations.
- Ensure the question does not combine multiple facts in a confusing or convoluted manner.
'''

    return '''
- The question should be clearly worded, concise, and directly related to descriptive information in the chunk.
- Avoid overly complex or verbose constructions.
- Penalize hypotheticals or any formulation that distracts from core informative intent.
'''



def get_groundedness_conditions(question_type):
    if question_type == question_types[0]:  # Yes-No
        return '''
- The answer must be directly supported by statements found in the document chunk.
- Do not reward generic or assumed responses — ensure the Yes/No conclusion is justified by explicit evidence.
- Penalize hallucinated conclusions or reasoning not grounded in the document.
'''

    elif question_type == question_types[1]:  # Yes-No w Condition
        return '''
- Both the Yes/No answer and the associated condition must be clearly supported by the document chunk.
- If either the outcome or the condition is not mentioned or implied in the context, penalize the answer.
- Avoid inferred or imagined conditions that are not explicitly grounded in the chunk.
'''

    elif question_type == question_types[2]:  # Legal-Obligation
        return '''
- The answer must be explicitly supported by legal obligations stated in the document.
- Penalize if it introduces legal interpretations, duties, or entities not present in the chunk.
- Do not reward answers that generalize or assume legal requirements beyond what's written.
'''

    elif question_type == question_types[3]:  # Factual
        return '''
- The answer must be factually accurate and verifiable from the chunk.
- Penalize inclusion of figures, names, or claims not found in the document.
- No extrapolation or assumptions — only grounded, stated facts should be used.
'''

    # Descriptive
    return ''' 
- The answer should describe information that is present or implied in the document chunk.
- Descriptions must remain faithful to the tone and scope of the original content.
- Penalize any exaggerations, invented attributes, or unsupported elaborations.
'''

def additional_instructions(metric_type, question_type):

    if metric_type == metric_types[0]:
        return get_accuracy_conditions(question_type)
    elif metric_type == metric_types[1]:
        return get_completeness_conditions(question_type)
    elif metric_type == metric_types[2]:
        return get_groundedness_conditions(question_type)

    return get_intent_conditions(question_type)


def get_prompt(chunk, question, answer, context, conditions, metric_type, question_type, title, summary, program_type, sector_type, incentive_amount_data):
    additional_prompts = additional_instructions(metric_type, question_type)
    
    # Entire formatting block for accuracy, completeness, intent, relevance, groundedness
    # You have already provided this in your code — just include all `if metric_type == ...` blocks as-is
    
    return res

def return_prompt(title, summary, program_type, sector_type, question_type, questions, incentive_amount_data):
    prompt = f"""You are an expert assistant..."""  # Full multi-question prompt template from original code
    for i, q in enumerate(questions, 1):
        prompt += f"{i}. {q}\n"
    prompt += """
---
Rate each question on a scale...
"""
    return prompt.strip()
