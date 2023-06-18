import pdfplumber
import os
from dotenv import load_dotenv
import jieba

jieba.initialize()
load_dotenv('.env')

# Define file path
cn_file_path = os.environ['INPUT_CN_FILE_PATH']


# Function to check if a sentence contains Chinese characters
def is_chinese(sentence):
    words = jieba.lcut(sentence)
    chinese_words = [word for word in words if word.isalpha()]
    return len(chinese_words) / len(words) > 0.5


def extract_text_from_pdf(file_path):
    with pdfplumber.open(file_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    return text


def save_text_to_file(text, output_file_path):
    text = text.replace('\n', ' ').replace('\r', '')
    with open(output_file_path, "w", encoding="utf-8") as file:
        file.write(text)


def get_sentences_segments(extracted_text):

    extracted_text = extracted_text.replace('\n', ' ').replace('\r', '')
    sentences = []
    sentence = ""
    for word in jieba.cut(extracted_text):
        sentence += word
        # if '。' in word:
        if '。' in word or '.' in word:
            sentences.append(sentence)
            sentence = ""
    return sentences


def save_segments_to_file(chinese_sentences):
    # for sentence in chinese_sentences:
    #     print(sentence+'\n')
    # Provide the path to the output text file
    output_file_path = os.environ["OUT_CN_FILE_PATH"]

    with open(output_file_path, 'w', encoding='utf-8') as file:
        for sentence in chinese_sentences:
            file.write(sentence + '\n\n')
            # print(sentence)

    # Print the extracted result
    print("Text extracted from PDF and saved to", output_file_path)
    print("Count Sentences", len(chinese_sentences))


textFromPdf = extract_text_from_pdf(cn_file_path)
# save_text_to_file(textFromPdf,"cn.txt")
segments = get_sentences_segments(textFromPdf)
# segments = get_sentences_segments("")
save_segments_to_file(segments)
