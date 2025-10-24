
from pypdf import PdfReader 
from text_cleaner import text_cleaner
import os
import getpass
from langchain.chat_models import init_chat_model
import fitz,base64,time,threading

if not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for Google Gemini: ")

llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai", temperature=0)



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

def extract_pdf_with_llm(file_path, output_path):
    cleaned_text = ""
    doc = fitz.open(file_path)
    total_pages = len(doc)
    
    # PDF-level timer
    pdf_start_time = time.time()
    pdf_timer_running = True
    
    def print_pdf_timer():
        while pdf_timer_running:
            elapsed = time.time() - pdf_start_time
            print(f"\r  📄 PDF Total Time: {elapsed:.1f}s", end='', flush=True)
            time.sleep(1)
    
    pdf_timer_thread = threading.Thread(target=print_pdf_timer, daemon=True)
    pdf_timer_thread.start()
    
    try:
        for page_num, page in enumerate(doc):
            print(f"\n\n  Processing Page {page_num + 1}/{total_pages}")
            
            # Page-level timer
            page_start_time = time.time()
            page_timer_running = True
            
            def print_page_timer():
                while page_timer_running:
                    elapsed = time.time() - page_start_time
                    print(f"\r    ⏱️  Page {page_num + 1} Time: {elapsed:.1f}s", end='', flush=True)
                    time.sleep(1)
            
            page_timer_thread = threading.Thread(target=print_page_timer, daemon=True)
            page_timer_thread.start()
            
            try:
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # High resolution
                img_bytes = pix.tobytes("png")
                img_base64 = base64.b64encode(img_bytes).decode("utf-8")
                img_data_uri = f"data:image/png;base64,{img_base64}"
                response = llm.invoke([
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"Analyze Page {page_num}. Extract and summarize all visible text and describe any images."},
                            {"type": "image_url", "image_url": {"url": img_data_uri}},
                        ],
                    }
                ])
                cleaned_text += f"--- Page {page_num} ---\n"
                cleaned_text += response.content + "\n\n"
            finally:
                page_timer_running = False
                time.sleep(0.1)
                page_elapsed = time.time() - page_start_time
                print(f"\r    ✅ Page {page_num + 1} completed in {page_elapsed:.2f}s")
    finally:
        pdf_timer_running = False
        time.sleep(0.1)
        pdf_elapsed = time.time() - pdf_start_time
        print(f"\n\n  ✅ PDF processing completed in {pdf_elapsed:.2f}s")
    
    if output_path:
        with open(output_path, "w") as f:
            f.write(cleaned_text)
        print(f'  💾 Cleaned text saved to: {output_path}')
    return cleaned_text

def clean_all_pdfs_in_folder_llm(folder_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            print(f"Cleaning file with LLM: {filename}")
            pdf_path = os.path.join(folder_path, filename)
            output_path = os.path.join(output_folder, f"cleaned_{filename}.txt")
            extract_pdf_with_llm(pdf_path, output_path)

if __name__ == "__main__":
    # paper_folder = "papers"
    # output_folder = "cleaned_papers"
    # clean_all_pdfs_in_folder(paper_folder, output_folder)
    file_path = "papers/Software-Engineering_Design_Patterns_for_Machine_Learning_Applications.pdf"
    print('Cleaning file:', file_path)
    output_path = "cleaned_papers/cleaned_Software-Engineering_Design_Patterns_for_Machine_Learning_Applications.pdf.txt"
    extract_pdf_with_llm(file_path, output_path)