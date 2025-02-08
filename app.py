import nltk
import stanza
import advertools as adv
import streamlit as st
from langdetect import detect
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import re

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

stanza.download('en')
stanza.download('hi')
stanza.download('ta')
stanza.download('te')
stanza.download('ur')

languages = {'hindi': 'hi', 'tamil': 'ta', 'telugu': 'te', 'urdu': 'ur'}
languages_code = {1: 'English', 2: 'Telugu', 3: 'Hindi', 4: 'Tamil', 5: 'Urdu'}

def get_nlp_pipeline(language):
    if language in languages:
        return stanza.Pipeline(lang=languages[language], processors='tokenize,pos,lemma', use_gpu=False)
    return None

language_suffixes = {
    "telugu": ["గా", "ను", "కి", "లో", "మీద"],
    "hindi": ["ने", "ता", "ही", "से", "को"],
    "tamil": ["ஆன்", "இன்", "உம்", "க்கு", "ல்"],
    "urdu": ["نے", "گا", "گی", "کا", "کی"],
}

def tokenize(text):
    return word_tokenize(text)

def pos_tagging(text, language):
    if language == 'english':
        tokens = word_tokenize(text)
        return pos_tag(tokens)
    
    nlp = get_nlp_pipeline(language)
    if nlp:
        doc = nlp(text)
        return [(word.text, word.pos) for sentence in doc.sentences for word in sentence.words]
    
    return []

def change_case(text, language):
    return (text.lower(), text.upper()) if language == 'english' else "Case change not supported."

def remove_punctuations(text):
    return re.sub(r'[^\u0C00-\u0C7F\u0900-\u097F\u0600-\u06FF\u0B80-\u0BFFa-zA-Z\s]', '', text)

def remove_stopwords(text, language):
    words = text.split()
    try:
        stop_words = stopwords.words('english') if language == 'english' else adv.stopwords.get(language, [])
    except:
        stop_words = []
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
    
    nlp = get_nlp_pipeline(language)
    if nlp:
        doc = nlp(text)
        return [word.lemma for sent in doc.sentences for word in sent.words]
    
    return text.split()

st.title("Multilingual Text Processor")
st.write("A simple NLP tool for text processing in multiple languages.")

language_option = st.selectbox("Select Language", list(languages_code.values()))
text_input = st.text_area("Enter your text")

technique_option = st.selectbox("Select Processing Technique", [
    "Tokenization", "POS Tagging", "Case Change", "Punctuation Removal",
    "Stopwords Removal", "Stemming", "Lemmatization"
])

if st.button("Process Text"):
    if text_input:
        language = language_option.lower()
        detected_language = detect(text_input)
        detected_language_full = next((lang for lang, code in languages.items() if code == detected_language), 'english')

        if detected_language_full != language:
            st.error(f"Text detected as {detected_language_full.capitalize()}, but you selected {language.capitalize()}.")
        else:
            if technique_option == "Tokenization":
                result = tokenize(text_input)
            elif technique_option == "POS Tagging":
                result = pos_tagging(text_input, language)
            elif technique_option == "Case Change":
                result = change_case(text_input, language)
            elif technique_option == "Punctuation Removal":
                result = remove_punctuations(text_input)
            elif technique_option == "Stopwords Removal":
                result = remove_stopwords(text_input, language)
            elif technique_option == "Stemming":
                result = stemming(text_input, language)
            elif technique_option == "Lemmatization":
                result = lemmatization(text_input, language)
            
            st.subheader(technique_option)
            st.write(result)
    else:
        st.warning("Please enter text to process.")

st.write("Made with ❤️ by [Abhinavtej Reddy](https://abhinavtejreddy.me)")