import json
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import wordnet
from itertools import tee, islice, chain
import nltk

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
    pos_tags = pos_tag(tokens)
    for token, tag in pos_tags:
        if re.match(r'[a-zA-Z]+', token):
            wordnet_pos = get_wordnet_pos(tag)
            lemmatized_tokens.append(lemmatizer_en.lemmatize(token, pos=wordnet_pos))
        else:
            lemmatized_tokens.append(token)
    return lemmatized_tokens

def generate_ngrams(tokens, n=2):
    return [' '.join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

def preprocess_text(text, ngram_n=2):
    if not isinstance(text, str):
        return []
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token not in stop_words_en]
    tokens = lemmatize_text(tokens)
    ngrams = generate_ngrams(tokens, ngram_n)
    return tokens + ngrams

def process_json(input_file, output_file, ngram_n=2):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    processed_data = []
    for item in data:
        processed_item = {}
        for key, value in item.items():
            if isinstance(value, str):
                processed_item[key] = preprocess_text(value, ngram_n)
            else:
                processed_item[key] = value
        processed_data.append(processed_item)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=4)

process_json("output.json", "processed_file.json", ngram_n=2)
                