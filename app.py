import nltk
import stanza
import advertools as adv
import streamlit as st
from langdetect import detect
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import re

# Download required resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')

stanza.download('en')
stanza.download('hi')
stanza.download('ta')
stanza.download('te')
stanza.download('ur')

# Language settings
languages = {'hindi': 'hi', 'tamil': 'ta', 'telugu': 'te', 'urdu': 'ur'}
languages_code = {1: 'English', 2: 'Telugu', 3: 'Hindi', 4: 'Tamil', 5: 'Urdu'}
nlp_pipelines = {lang: stanza.Pipeline(code, processors='tokenize,pos,lemma') for lang, code in languages.items()}

language_suffixes = {
    "telugu": ["గా", "ను", "కి", "లో", "మీద"],
    "hindi": ["ने", "ता", "ही", "से", "को"],
    "tamil": ["ஆன்", "இன்", "உம்", "க்கு", "ல்"],
    "urdu": ["نے", "گا", "گی", "کا", "کی"],
}

# Text Processing Functions
def tokenize(text):
    return nltk.word_tokenize(text)

def change_case(text, language):
    return (text.lower(), text.upper()) if language == 'english' else "Case change not supported."

def remove_punctuations(text):
    return re.sub(r'[^\u0C00-\u0C7F\u0900-\u097F\u0600-\u06FF\u0B80-\u0BFFa-zA-Z\s]', '', text)

def remove_stopwords(text, language):
    words = text.split()
    stop_words = stopwords.words('english') if language == 'english' else adv.stopwords.get(language, [])
    return ' '.join([word for word in words if word.lower() not in stop_words])

def stemming(text, language):
    words = text.split()
    if language == 'english':
        return [PorterStemmer().stem(word) for word in words]
    return [stem_language_word(word, language) for word in words]

def stem_language_word(word, language):
    for suffix in language_suffixes.get(language.lower(), []):
        if word.endswith(suffix):
            return word[:-len(suffix)]
    return word

def lemmatization(text, language):
    if language == 'english':
        return [WordNetLemmatizer().lemmatize(word) for word in text.split()]
    doc = nlp_pipelines[language](text)
    return [word.lemma for sent in doc.sentences for word in sent.words]

# Streamlit UI
st.title("Multilingual Text Processor")
st.write("A simple NLP tool for text processing in multiple languages.")

language_option = st.selectbox("Select Language", list(languages_code.values()))
text_input = st.text_area("Enter your text")

if st.button("Process Text"):
    if text_input:
        language = language_option.lower()
        detected_language = detect(text_input)
        
        if detected_language[:2] != language[:2]:
            st.error("Text doesn't match the selected language.")
        else:
            processed_data = {
                "Tokenized": tokenize(text_input),
                "Case Changed": change_case(text_input, language),
                "Punctuation Removed": remove_punctuations(text_input),
                "Stopwords Removed": remove_stopwords(text_input, language),
                "Stemming": stemming(text_input, language),
                "Lemmatization": lemmatization(text_input, language),
            }
            
            for key, value in processed_data.items():
                st.subheader(key)
                st.write(value)
    else:
        st.warning("Please enter text to process.")
