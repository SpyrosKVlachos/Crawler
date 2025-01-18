import json
from sklearn.metrics import precision_score, recall_score, f1_score

with open('output.json', 'r', encoding='utf-8') as f:
    documents = {str(doc['id']): doc for doc in json.load(f)}

with open('questions.json', 'r', encoding='utf-8') as f:
    questions = json.load(f)['questions']

with open('relevance.json', 'r', encoding='utf-8') as f:
    relevance = json.load(f)['relevance']

with open("inverted_index.json", "r", encoding="utf-8") as f:
    inverted_index = json.load(f)

def search(query, inverted_index):
    terms = query.lower().split()
    result_set = [set(inverted_index.get(term, [])) for term in terms]
    if result_set:
        return list(set.intersection(*result_set))
    return []

def calculate_metrics(retrieved, ground_truth, all_doc_ids):
    all_precisions = []
    all_recalls = []
    all_f1s = []
    map_score = 0

    for q_id, relevant_docs in ground_truth.items():
        retrieved_docs = set(retrieved.get(q_id, []))
        relevant_docs = set(relevant_docs)

        binary_retrieved = [1 if doc in retrieved_docs else 0 for doc in all_doc_ids]
        binary_relevant = [1 if doc in relevant_docs else 0 for doc in all_doc_ids]

        precision = precision_score(binary_relevant, binary_retrieved, zero_division=0)
        recall = recall_score(binary_relevant, binary_retrieved, zero_division=0)
        f1 = f1_score(binary_relevant, binary_retrieved, zero_division=0)

        all_precisions.append(precision)
        all_recalls.append(recall)
        all_f1s.append(f1)

        if relevant_docs:
            precision_at_k = [
                len(relevant_docs & set(retrieved_docs[:k + 1])) / (k + 1)
                for k in range(len(retrieved_docs))
            ]
            avg_precision = sum(precision_at_k) / len(relevant_docs) if relevant_docs else 0
            map_score += avg_precision
    
    mean_precision = sum(all_precisions) / len(all_precisions) if all_precisions else 0
    mean_recall = sum(all_recalls) / len(all_recalls) if all_recalls else 0
    mean_f1 = sum(all_f1s) / len(all_f1s) if all_f1s else 0
    map_score /= len(ground_truth) if ground_truth else 1

    return mean_precision, mean_recall, mean_f1, map_score

all_doc_ids = set(documents.keys())

retrieved_documents = {}
for q_id, query in questions.items():
    retrieved_documents[q_id] = search(query, inverted_index)
    print(f"Ερώτηση: {query}")
    print(f"Συναφή έγγραφα: {retrieved_documents[q_id]}")

precision, recall, f1, map_score = calculate_metrics(retrieved_documents, relevance, all_doc_ids)

print(f"Ακρίβεια: {precision:.2f}")
print(f"Ανάκληση: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
print(f"MAP: {map_score:.2f}")