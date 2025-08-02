from .config import context_df, question_types, policy_types

def return_examples(cnt, question_type, program_type):
    """
    Samples high-rated examples from the context_df for few-shot learning.
    """
    temp_df = context_df[
        (context_df['policy_type'] == program_type) &
        (context_df['question_type'] == question_type)
    ].sample(n=cnt)

    temp_df = temp_df[['question', 'answer', 'condition', 'context', 'q_rating', 'a_rating']]
    temp_df['rating'] = temp_df.apply(lambda row: 0.5 * row['q_rating'] + 0.5 * row['a_rating'], axis=1)
    temp_df = temp_df.sort_values(by='rating', ascending=False).to_numpy()

    data = []
    for i in range(cnt):
        data.append({
            'question': temp_df[i][0],
            'answer': temp_df[i][1],
            'conditions': temp_df[i][2],
            'context': temp_df[i][3]
        })
    
    return f'{data}'


def return_prompt(title, state, program_type, sector_type, text, question_type, fewshot_examples, num_questions):
    """
    Constructs the final prompt string by combining instructions, metadata, examples, and the text.
    """
    example_injection_prompts = {
        (question_types[0], policy_types[1]): """
- Generate a Yes/No question based on an Incentives policy document. Only create questions with Yes/No answers and doesn't require any condition.
- These questions should confirm or deny specific eligibility, benefits, conditions, or scenarios related to the incentive.
- Examples include asking whether a specific benefit is available or whether a certain group qualifies.
        """,
        (question_types[0], policy_types[0]): """
- Generate a Yes/No question based on a Regulatory policy document. Only create questions with Yes/No answer and doesn't require any condition.
- These questions should verify the presence or absence of rules, requirements, enforcement dates, or compliance expectations. 
- The question should be answerable directly with 'Yes' or 'No' based on the policy text.
        """,
        (question_types[1], policy_types[1]): """
- Generate a Yes/No question with some conditions based on an Incentives policy document.
- Only create questions that require some assumptions from the text to be true. Include those assumptions in the list format.
- Do not create questions that don't require any other condition.
- These questions should confirm or deny specific eligibility, benefits, conditions, or scenarios related to the incentive. 
        """,
        (question_types[1], policy_types[0]): """
- Generate a Yes/No question with some conditions based on an Regulatory policy document.
- Only create questions that require some assumptions from the text to be true. Include those assumptions in the list format.
- Do not create questions that don't require any other condition.
- These questions should verify the presence or absence of rules, requirements, enforcement dates, or compliance expectations. 
        """,
        (question_types[3], policy_types[1]): """
- Generate a factual question that seeks concrete details from an Incentives policy document. 
- This can include information such as benefit amounts, deadlines, scheme names, or eligibility thresholds. 
- The answer should be explicitly stated in the document.
        """,
        (question_types[3], policy_types[0]): """
- Generate a factual question that extracts specific details from a Regulatory policy document. 
- This may include enforcement dates, penalties, defined terms, or mandatory actions. 
- The answer should be a direct fact mentioned in the policy.
        """,
        (question_types[4], policy_types[1]): """
- Generate a descriptive question that asks for an explanation or elaboration on the Incentives policy. 
- This may involve describing the purpose of the incentive, how it operates, or the broader goals it serves. 
- The answer should summarize or paraphrase parts of the policy in a detailed manner.
        """,
        (question_types[4], policy_types[0]): """
- Generate a descriptive question that prompts an explanation of the Regulatory policy. 
- This could include the reasoning behind the regulation, the mechanisms of enforcement, or the sectors it targets. 
- The answer should provide a clear and informative overview derived from the document.
        """,
        (question_types[2], policy_types[1]): """
- Generate a question that identifies any legal duties or formal commitments required in an Incentives policy document. 
- This may involve mandatory documentation, procedural requirements, or compliance terms for receiving the incentive.
        """,
        (question_types[2], policy_types[0]): """
- Generate a question that reveals the legal obligations imposed by a Regulatory policy. 
- This includes rules that must be followed, penalties for non-compliance, reporting duties, or formal restrictions.
        """,
    }

    additional_instruction = example_injection_prompts.get((question_type, program_type), "")
    examples = return_examples(fewshot_examples, question_type, program_type)

    return rf'''You are a QA dataset generator designed to create high-quality question-answer-context from solar and energy policy documents. Follow the instructions carefully.

## Instructions:
- The text is US solar policy document in markdown format. For questions and answers, modify the text to make it readable and user-friendly.
- Extract key findings that are valuable and avoid unnecessary or repetitive details.
- Ensure the answers are present in the text. Do not make any inferences or provide answers beyond what is mentioned.
- Generate at most {num_questions} questions, answers, and contexts. Answer can be modified from the text but context should be exact same words from the text.
- Capture all important information in distinct QA pairs, avoiding any overlap or redundancy between pairs.
- The text contains structured data like tables and lists, understand the information given and create questions from that.
- The text has some code for html, markdown code, separators, blank lines which is unnecessary for the original text. So, remove it from the questions and answers.
- The text has many order lists. So, there are many letter like a, b, etc at the start and at the end showing the order number of that list. Remove them from the answers. 
- The question should be of type {question_type}. Do not create questions of other types. There are total 5 types - Yes/No (without conditions), Yes/No with conditions, Legal Obligation, Factual and Descriptive.
- The questions should contain location information - {state} and the policy title - {title} and make the questions as specific as possible.
- Add detailed scenarios in the questions if possible. Use Program Sector - {sector_type} to create first person scenarios for users who want to know more about the policy.
- Generate scenarios that can change the outcome of the question based on the user requirements.
- Do not use keywords like this program or sections reference that require a specific document to look and answer the question.
- Never create questions from the metadata entirely. Only use it to make questions more rich and specific.
- The text may have reference of other sections which needs to be removed in the answers to remove confusions.
- The questions should be of {question_type} type only. It shouldn't be of any other type no matter what.
- If there are no questions possible from the text. Return 'NA'. Do not return anything else, no matter what.
- Use examples given below to understand the problem in a better way and improve your final result. But do not create questions from the examples.

## Additional Instructions for question type {question_type}:
{additional_instruction}

## Examples:
{examples}

## Metadata (for reference only):
**title** - {title}
**state** - {state}
**Program Type** - {program_type}
**Program Sector** - {sector_type}

## Output Format:
### <QA Pair number> 
**Question** - <question>
**Answer** - <answer>
**Conditions** - [<conditions>]
**Context** - <context>

## Text:
{text}
'''
