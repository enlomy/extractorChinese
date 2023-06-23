from pdfminer.high_level import extract_text
import nltk
import os
import re

from dotenv import load_dotenv
from nltk.tokenize import sent_tokenize
from langdetect import detect

load_dotenv('.env')
# Define file path
file_path = os.environ['INPUT_EN_FILE_PATH']


# Function to check if a sentence contains English words
def is_english(sentence):
    words = nltk.word_tokenize(sentence)
    english_words = [word for word in words if word.isalpha()]
    if len(words) == 0: 
        return False
    return len(english_words) / len(words) > 0.5


def extract_text_from_pdf(file_path):
    text = extract_text(file_path)
    return text


def save_text_to_file(text, output_file_path):
    with open(output_file_path, "w", encoding="utf-8") as file:
        file.write(text)


def get_sentences_segments(extracted_text):
    extracted_text = re.sub('[“”]', '"', extracted_text)
    # Segmentation of sentences
    sentences = extracted_text.split('\n\n')
    english_sentences = [sentence for sentence in sentences if sentence and is_english(sentence) and detect(sentence) == 'en']
    return english_sentences


def save_segments_to_file(english_sentences):
    # Provide the path to the output text file
    output_file_path = os.environ["OUT_EN_FILE_PATH"]

    with open(output_file_path, 'w',encoding="utf-8") as file:
        for sentence in english_sentences:
            sentence = sentence.replace('\n', ' ').replace('\r', '')
            file.write(sentence + "\n\n")

    # Print the extracted result
    print("Text extracted from PDF and saved to", output_file_path)
    print("Count Sentences", len(english_sentences))


textFromPdf = extract_text_from_pdf(file_path)
segments = get_sentences_segments(textFromPdf)
save_segments_to_file(segments)
