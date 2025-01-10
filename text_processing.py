import json
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.util import ngrams

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('owm-1.4')
lemmatizer = WordNetLemmatizer()
stop_words_en = set(stopwords.words('english'))
stop_words_el = set(stopwords.words('greek'))
stop_words = stop_words_en.union(stop_words_el)

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[\w\s+]', ' ', text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens

def number_convention(tokens):
    new_tokens = []
    for token in tokens:
        try:
            int(token)
            new_tokens.append("<NUMBER>")
        except ValueError:
            new_tokens.append(token)
    return new_tokens

def process_json(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    processed_data = []
    for item in data:
        processed_item = {}
        for key, value in item.items():
            if key in ["title", "content"]:
                tokens = preprocess_text(value)
                tokens = number_convention(tokens)
                bigrams = list(ngrams(tokens, 2))
                processed_item[key] = {"tokens": tokens, "bigrams": bigrams}
            else:
                processed_item[key] = value
        processed_data.append(processed_item)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=4)

process_json("output.json", "processed_file.json")