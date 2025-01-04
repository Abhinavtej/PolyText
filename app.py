import nltk
import os
import stanza
from flask import Flask, render_template, request
from langdetect import detect
import re
import advertools as adv
from nltk.corpus import stopwords as s_w
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)

# Define static paths for resource files
NLTK_DATA_PATH = os.path.join(app.root_path, 'static', 'nltk_data')
STANZA_RESOURCES_PATH = os.path.join(app.root_path, 'static', 'stanza_resources')

# Configure NLTK to use the static path
nltk.data.path.append(NLTK_DATA_PATH)

# Setup Stanza to use the static path
stanza.download_dir = STANZA_RESOURCES_PATH

# Setup for languages
languages = {'hindi': 'hi', 'tamil': 'ta', 'telugu': 'te', 'urdu': 'ur'}
languages_code = {1: 'English', 2: 'Telugu', 3: 'Hindi', 4: 'Tamil', 5: 'Urdu'}

# Initialize NLP pipelines
nlp_pipelines = {}
for lang, code in languages.items():
    nlp_pipelines[lang] = stanza.Pipeline(code, dir=STANZA_RESOURCES_PATH, processors='tokenize,pos,lemma')

language_suffixes = {
    "telugu": ["గా", "ను", "కి", "లో", "మీద"],
    "hindi": ["ने", "ता", "ही", "से", "को"],
    "tamil": ["ஆன்", "இன்", "உம்", "க்கு", "ல்"],
    "urdu": ["نے", "گا", "گی", "کا", "کی"],
}

# Functions
def tokenize(text):
    return nltk.word_tokenize(text)

def change_case(text, language):
    if language == 'english':
        return text.lower(), text.upper()
    else:
        return f"Case change not supported for {language.upper()}."
    
def remove_punctuations(text):
    return re.sub(r'[^\u0C00-\u0C7F\u0900-\u097F\u0600-\u06FF\u0B80-\u0BFFa-zA-Z\s]', '', text)

def remove_stopwords(text, language):
    words = text.split()
    if language == 'english':
        stop_words = s_w.words('english')
    else:
        stop_words = sorted(adv.stopwords[language])
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)

def stemming(text, language):
    words = text.split()
    if language == 'english':
        stemmer = PorterStemmer()
        return [stemmer.stem(word) for word in words]
    return [stem_language_word(word, language) for word in words]

def stem_language_word(word, language):
    suffixes = language_suffixes.get(language.lower(), [])
    for suffix in suffixes:
        if word.endswith(suffix):
            return word[:-len(suffix)]
    return word

def lemmatization(text, language):
    if language == 'english':
        lemmatizer = WordNetLemmatizer()
        words = text.split()
        return [lemmatizer.lemmatize(word) for word in words]
    doc = nlp_pipelines[language](text)
    return [word.lemma for sent in doc.sentences for word in sent.words]

# Routes
@app.route('/')
def index():
    return render_template('index.html', languages=languages_code)

@app.route('/process', methods=['POST'])
def process():
    try:
        language_num = int(request.form['language'])
        language = languages_code.get(language_num)
        
        if not language:
            return render_template('index.html', error="Invalid language selection.", languages=languages_code)

        text = request.form['text']
        language = language.lower()
        dl = language[:2]

        detected_language = detect(text)
        if dl != detected_language:
            return render_template('index.html', error="Text doesn't match the selected language.", languages=languages_code)

        processed_data = {
            "tokenized": tokenize(text),
            "case_changed": change_case(text, language),
            "punctuation_removed": remove_punctuations(text),
            "stopwords_removed": remove_stopwords(text, language),
            "stemming": stemming(text, language),
            "lemmatization": lemmatization(text, language),
        }

        return render_template('result.html', data=processed_data)

    except KeyError as e:
        return render_template('index.html', error="Invalid language selection.", languages=languages_code)
    except Exception as e:
        return render_template('index.html', error=f"An error occurred: {str(e)}", languages=languages_code)


if __name__ == '__main__':
    if not os.path.exists(os.path.join(NLTK_DATA_PATH, 'tokenizers/punkt')):
        raise FileNotFoundError(f"'punkt' resource not found in {NLTK_DATA_PATH}.")
    if not os.path.exists(os.path.join(NLTK_DATA_PATH, 'corpora/stopwords')):
        raise FileNotFoundError(f"'stopwords' resource not found in {NLTK_DATA_PATH}.")
    port = int(os.environ.get('PORT', 3001))
    app.run(debug=True, host='0.0.0.0', port=port)
