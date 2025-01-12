import re
import json
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import wordnet
import nltk
from ranking import boolean_retrieval, tfidf_retrieval, vsm_retrieval, bm25_retrieval

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
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
    lemmatized_tokens = []
    for token in tokens:
        pos = get_wordnet_pos(pos_tag([token])[0][1])
        lemmatized_tokens.append(lemmatizer_en.lemmatize(token, pos=pos))
    return lemmatized_tokens
    
def preprocess_query(query):
    tokens = word_tokenize(query)
    processed_tokens = []

    for token in tokens:
        cleaned_token = re.sub(r'[^\w\s]', '', token.lower())
        if cleaned_token and cleaned_token not in stop_words_en:
            lemmatized_token = lemmatize_text([cleaned_token])[0]
            processed_tokens.append(lemmatized_token)
        
    return ' '.join(processed_tokens)

def load_output_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)
    
def display_results(results, output_data):
    for doc_id in results:
        doc = next((item for item in output_data if item['id'] == doc_id), None)
        if doc:
            print(f"Έγγραφο {doc_id}: Τίτλος: {doc['title']}, URL: {doc['url']}")
        else:
            print(f"Εγγραφο {doc_id}: Δεν βρέθηκαν πληροφορίες")

def main():
    processed_file_path = "processed_file.json"
    output_file_path = "output.json"

    output_data = load_output_file(output_file_path)
    data = {item["id"]: item["content"] for item in output_data}

    boolean = boolean_retrieval(processed_file_path)
    tfidf = tfidf_retrieval(processed_file_path)
    vsm = vsm_retrieval(processed_file_path)
    bm25 = bm25_retrieval(processed_file_path)

    while True:
        query = input("Εισάγεται το ερώτημα αναζήτησης (exit για έξοδο): ").strip()
        if query.lower == "exit":
            break

        search_type = input("Επιλέξτε τύπο αναζήτησης (1: Boolean, 2: TF-IDF, 3: VSM, 4: BM25): ").strip()
        processed_query = preprocess_query(query)

        if search_type == "1":
            results = boolean.search(processed_query)
        elif search_type == "2":
            results = tfidf.search(processed_query)
        elif search_type == "3":
            results = vsm.search(processed_query, data)
        elif search_type == "4":
            results = bm25.search(processed_query, data)
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