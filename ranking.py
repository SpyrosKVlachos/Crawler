import re
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi

def boolean_retrieval(query, documents):
    query = query.strip().lower()
    tokens = re.split(r'\s+(and|or|not)\s+', query)

    terms = [t for t in tokens if t not in {"and", "or", "not"}]
    operators = [t for t in tokens if t in {"and", "or", "not"}]

    term_sets = []
    for term in terms:
        term_set = {doc["id"] for doc in documents if term in doc["content"].lower()}
        term_sets.append(term_set)

    result_set = term_sets[0]
    for i, operator in enumerate(operators):
        if operator == "and":
            result_set &= term_sets[i + 1]
        elif operator == "or":
            result_set |= term_sets[i + 1]
        elif operator == "not":
            result_set -= term_sets[i + 1]
    
    filtered_documents = [doc for doc in documents if doc["id"] in result_set]
    return filtered_documents

def tfidf_retrieval(query, documents):
    corpus = [doc["content"] for doc in documents]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()

    ranked_docs = sorted(
        zip(documents, similarities),
        key=lambda x: x[1],
        reverse=True
    )
    return [doc for doc, score in ranked_docs if score > 0]

def vsm_retrieval(query, documents):
    return tfidf_retrieval(query, documents)

def bm25_retrieval(query, documents):
    tokenized_corpus = [doc["content"].lower().split() for doc in documents]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    ranked_docs = sorted(
        zip(documents, scores),
        key=lambda x: x[1],
        reverse=True
    )
    return [doc for doc, score in ranked_docs if score > 0]