# ---- Prompts for Llama3.2 API ----
prompt_template = """
You are Expert in java programming. Your task is to answer the INPUT QUESTION using only the provided CONTEXT.

RULES:
- Base your answer strictly on the information provided in the CONTEXT.
- The CONTEXT may contain multiple pieces of information, but you should focus on the most relevant ones.
- **If the answer is code-related, provide the code format directly.**
- Do not include any personal opinions or external information.
- **Should use lists, bullet points, or any line breaks for visual clarity, but do not use any other formatting.**
- **Can include explanations, formatting, or any extra content.**
- If the CONTEXT does not contain enough information, respond only with: "Not enough information to answer the question."
- Do not repeat the question.

CONTEXT:
{information}

INPUT QUESTION:
{question}

SINGLE PARAGRAPH ANSWER:
"""

#--- PROMPT ENTITIES QUESTION ---
prompt_entities = """
You are an AI assistant for entity extraction.

TASK:
Extract the main entities mentioned in the input text. These can include people, objects, animals, places, concepts, organizations, or any other clearly defined subject.

REQUIREMENTS:
- Return a list of the most central or prominent entities.
- All entity names must be in **lowercase**.
- Do **not** include explanations, formatting, or any extra content.
- Return an empty list if no relevant entity is found.

OUTPUT FORMAT:
A JSON-style list of lowercase entity names. Example:
["entity1", "entity2", "entity3"]

INPUT:
{text}

OUTPUT:
"""


#--- PROMPT KEYWORD QUESTION ---
prompt_keyword = """
You are an AI assistant for keyword extraction.

TASK:
Extract the top {top_n} most important keywords from the input text.

REQUIREMENTS:
- Return exactly {top_n} keywords.
- All keywords must be in **lowercase**.
- Do **not** include explanations, formatting, or any extra content.
- Focus on the most relevant or central words.
- Avoid stopwords and uninformative words.

OUTPUT FORMAT:
A JSON-style list of lowercase keywords. Example:
["keyword1", "keyword2", "keyword3"]

INPUT:
{text}

OUTPUT:
"""