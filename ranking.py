import re
import json
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import wordnet
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

lemmatizer_en = WordNetLemmatizer()
stop_words_en = set(stopwords.words('english'))

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def lemmatize_text(tokens):
    return [lemmatizer_en.lemmatize(token, pos=get_wordnet_pos(pos_tag([token])[0][1])) for token in tokens]

def preprocess_query(query):
    tokens = word_tokenize(query)
    processed_tokens = [
        lemmatize_text([re.sub(r'[^\w\s]', '', token.lower())])[0]
        for token in tokens
        if re.sub(r'[^\w\s]', '', token.lower()) and re.sub(r'[^\w\s]', '', token.lower()) not in stop_words_en
    ]
    return ' '.join(processed_tokens)

def load_output_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Σφάλμα: Το αρχείο {file_path} δεν βρέθηκε.")
        return None
    except json.JSONDecodeError:
        print(f"Σφάλμα: Το αρχείο {file_path} δεν είναι έγκυρο JSON.")
        return None

def display_results(results, output_data):
    if output_data is None:
        return

    for doc_id in results:
        doc = next((item for item in output_data if item['id'] == doc_id), None)
        if doc:
            print(f"Έγγραφο {doc_id}: Τίτλος: {doc['title']}, URL: {doc['url']}")
        else:
            print(f"Έγγραφο {doc_id}: Δεν βρέθηκαν πληροφορίες για το έγγραφο με ID {doc_id}")

def boolean_retrieval(query, documents):
    query = query.strip().lower()
    tokens = re.findall(r'\b\w+\b|and|or|not', query, flags=re.IGNORECASE)
    
    print(f"Tokens από το ερώτημα: {tokens}") 

    def evaluate(operators, operands):
        operator = operators.pop()
        if operator == "not":
            term = operands.pop()
            operands.append(doc_ids - term)
        else:
            right = operands.pop()
            left = operands.pop()
            if operator == "and":
                operands.append(left & right)
            elif operator == "or":
                operands.append(left | right)

    doc_ids = set(range(1, len(documents) + 1))
    terms = {doc["id"]: set(doc["content"].lower().split()) for doc in documents}

    operands = []
    operators = []
    precedence = {"not": 3, "and": 2, "or": 1}
    
    for token in tokens:
        if token in precedence:
            while (operators and 
                    precedence[operators[-1]] >= precedence[token]):
                evaluate(operators, operands)
            operators.append(token)
        else:
            term_set = {doc_id for doc_id, content in terms.items() if token in content}
            print(f"Όρος '{token}' βρέθηκε στα έγγραφα: {sorted(term_set)}")  
            operands.append(term_set)

    while operators:
        evaluate(operators, operands)

    return sorted(operands.pop() if operands else [])

def tfidf_retrieval(query, processed_documents):
    vectorizer = TfidfVectorizer()

    documents = [doc["content"] for doc in processed_documents]
    all_data = documents + [query]

    tfidf_matrix = vectorizer.fit_transform(all_data)

    query_vector = tfidf_matrix[-1]

    cosine_similarities = cosine_similarity(tfidf_matrix[:-1], query_vector.reshape(1, -1))

    doc_ids = [doc["id"] for doc in processed_documents]
    sorted_results = sorted(zip(doc_ids, cosine_similarities.squeeze()), key=lambda x: x[1], reverse=True)

    return [result[0] for result in sorted_results]

def vms_retrieval(query, processed_documents):
    vectorizer = TfidfVectorizer()

    documents = [doc["content"] for doc in processed_documents]
    all_data = documents + [query]

    tfidf_matrix = vectorizer.fit_transform(all_data)

    query_vector = tfidf_matrix[-1]

    cosine_similarities = cosine_similarity(tfidf_matrix[:-1], query_vector.reshape(1, -1))

    doc_ids = [doc["id"] for doc in processed_documents]
    sorted_results = sorted(zip(doc_ids, cosine_similarities.squeeze()), key=lambda x: x[1], reverse=True)

    return [result[0] for result in sorted_results if result[1] > 0]

def bm25_retrieval(query, processed_documents):
    corpus = [doc["content"].split() for doc in processed_documents]
    bm25 = BM25Okapi(corpus)
    tokenized_query = query.split()

    doc_scores = bm25.get_scores(tokenized_query)

    doc_ids = [doc["id"] for doc in processed_documents]
    sorted_results = sorted(zip(doc_ids, doc_scores), key=lambda x: x[1], reverse=True)

    return [result[0] for result in sorted_results if result[1] > 0]


def main():
    output_file_path = "output.json"
    output_data = load_output_file(output_file_path)

    if output_data is None:
        return

    processed_documents = [{"id": item["id"], "content": item["content"]} for item in output_data]

    while True:
        query = input("Εισάγεται το ερώτημα αναζήτησης (exit για έξοδο): ").strip()
        if query.lower() == "exit":
            break

        search_type = input("Επιλέξτε τύπο αναζήτησης (1: Boolean 2: TF-IDF 3: VMS 4: BM25): ").strip()

        if search_type == "1":
            results = boolean_retrieval(query, processed_documents)
        elif search_type == "2":
            results = tfidf_retrieval(query, processed_documents)
        elif search_type == "3":
            results = vms_retrieval(query, processed_documents)
        elif search_type == "4":
            results = bm25_retrieval(query, processed_documents)
        else:
            print("Μη έγκυρη επιλογή. Δοκιμάστε ξανά.")
            continue

        if results:
            print("Αποτελέσματα αναζήτησης:")
            display_results(results, output_data)
        else:
            print("Δεν βρέθηκαν έγγραφα που να ταιριάζουν.")

if __name__ == "__main__":
    main()
