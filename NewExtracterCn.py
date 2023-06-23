from pdfminer.high_level import extract_text
import os
from dotenv import load_dotenv
import jieba

jieba.initialize()
load_dotenv('.env')

# Define file path
cn_file_path = os.environ['INPUT_CN_FILE_PATH']

# Function to check if a sentence contains Chinese characters
def is_chinese(sentence):
    # Check if the sentences are chinese
    words = jieba.lcut(sentence)
    chinese_words = [word for word in words if word.isalpha()]
    return len(chinese_words) / len(words) > 0.5


def extract_text_from_pdf(file_path):
    # Extract as binary text from PDF
    text = extract_text(file_path)
    return text


def save_text_to_file(text, output_file_path):
    # Save text to file as normal
    text = text.replace('\n', ' ').replace('\r', '')
    with open(output_file_path, "w", encoding="utf-8") as file:
        file.write(text)


def get_sentences_segments(extracted_text):
    # Split sentences as segments
    sentences = extracted_text.split('\n\n')
    return sentences


def save_segments_to_file(chinese_sentences):
    # Provide the path to the output text file
    output_file_path = os.environ["OUT_CN_FILE_PATH"]

    with open(output_file_path, 'w', encoding='utf-8') as file:
        for sentence in chinese_sentences:
            sentence = sentence.replace('\n', ' ').replace('\r', '') # Remove Unnecessary linebreak
            file.write(sentence + '\n\n')

    # Print the extracted result
    print("Text extracted from PDF and saved to", output_file_path)
    print("Count Sentences", len(chinese_sentences))


textFromPdf = extract_text_from_pdf(cn_file_path)
segments = get_sentences_segments(textFromPdf)
save_segments_to_file(segments)
