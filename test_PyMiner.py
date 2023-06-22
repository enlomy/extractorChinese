from pdfminer.high_level import extract_text
import os
from dotenv import load_dotenv
load_dotenv('.env')

# Define file paths
input_file_path = os.environ['INPUT_EN_FILE_PATH']

# Open the PDF file in read-binary mode
text = extract_text(input_file_path)
with open('output.txt', 'w', encoding='utf-8') as output_file:
    output_file.write(text)