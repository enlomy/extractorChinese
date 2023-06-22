import PyPDF2
import os
from dotenv import load_dotenv
load_dotenv('.env')

# Define file paths
input_file_path = os.environ['INPUT_EN_FILE_PATH']
output_file_path = os.environ['OUTPUT_TEST_FILE_PATH']

# Open the PDF file in read-binary mode
with open(input_file_path, 'rb') as pdf_file:
    reader = PyPDF2.PdfReader(pdf_file)

    text = ''
    for page in reader.pages:
        text += page.extract_text()

    with open('output.txt', 'w', encoding='utf-8') as output_file:
        output_file.write(text)