from dotenv import load_dotenv
# Tìm đường dẫn tới file .env ở thư mục gốc
load_dotenv()
import spacy

_nlp = None

def get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp
 

def extract_keywords_from_question(question, top_n=5):
    nlp = get_nlp()
    doc = nlp(question)

    keywords = [
        token.lemma_.lower()
        for token in doc
        if token.pos_ in {"NOUN", "PROPN", "ADJ"}
        and not token.is_stop
        and token.is_alpha
    ]

    seen = set()
    keywords = [k for k in keywords if not (k in seen or seen.add(k))]

    return keywords[:top_n]


def extract_entities(question, top_n=5):
    nlp = get_nlp()
    doc = nlp(question)

    noun_phrases = []

    for chunk in doc.noun_chunks:
        text = chunk.text.lower().strip()

        if any(tok.is_stop for tok in chunk):
            continue

        if len(text) < 3:
            continue

        noun_phrases.append(text)

    # unique + giữ thứ tự
    seen = set()
    noun_phrases = [x for x in noun_phrases if not (x in seen or seen.add(x))]

    return noun_phrases[:top_n]


# --- Rerank contexts ---
def rerank_contexts_with_keywords(output_database, similarities, keywords, entities, question, weight=0.8, k=3):
    question_lower = question.lower()
    scores = []

    for i, chunk in enumerate(output_database):
        try:
            chunk_lower = chunk.lower()
            score = similarities[i] if i < len(similarities) else 0.0

            keyword_bonus = sum(1.0 for kw in keywords if kw.lower() in chunk_lower)

            # Only bonus if entity is relevant to question
            entity_bonus = 0.0
            for ent in entities:
                if ent in chunk_lower and ent in question_lower:
                    entity_bonus += 2.0

            final_score = score + weight * (keyword_bonus + entity_bonus)
            scores.append((final_score, i))
        except Exception:
            scores.append((0.0, i))

    scores.sort(reverse=True)
    return [i for _, i in scores[:k]]


def get_top_k_contexts(context_chunks, question, similarities, k=3):

    keywords = extract_keywords_from_question(question)
    entities = extract_entities(question)

    top_indices = rerank_contexts_with_keywords(context_chunks, similarities, keywords, entities, question)[:k]

    # for i in top_indices:
    #     print(f"- ({similarities[i]:.3f}) {context_chunks[i]}")
    return [context_chunks[i] for i in top_indices]


