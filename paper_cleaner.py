
from pypdf import PdfReader 
from text_cleaner import text_cleaner
import os

def read_pdf_file(file_path):
    text_content = ""
    with open(file_path, "rb") as file:
        reader = PdfReader(file)
        for page in reader.pages:
            text_content += page.extract_text() + "\n"
    
    return text_content

def clean_pdf_content(file_path):
    raw_text = read_pdf_file(file_path)
    cleaned_text = text_cleaner(raw_text)
    return cleaned_text

def save_cleaned_text(file_path, output_path):
    cleaned_text = clean_pdf_content(file_path)
    with open(output_path, "w") as file:
        file.write(cleaned_text)

def clean_all_pdfs_in_folder(folder_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            print(f"Cleaning file: {filename}")
            pdf_path = os.path.join(folder_path, filename)
            output_path = os.path.join(output_folder, f"cleaned_{filename}.txt")
            save_cleaned_text(pdf_path, output_path)
            print(f"Cleaned and saved: {output_path}")

if __name__ == "__main__":
    paper_folder = "papers"
    output_folder = "cleaned_papers"
    clean_all_pdfs_in_folder(paper_folder, output_folder)