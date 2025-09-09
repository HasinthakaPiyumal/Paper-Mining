import os
import getpass
from langchain.chat_models import init_chat_model
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence

if not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for Google Gemini: ")

llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai")

# Configurations for text splitting
# splitter = RecursiveCharacterTextSplitter(
#     chunk_size=3000,
#     chunk_overlap=200,
#     separators=["\n\n", "\n", " ", ""]
# )

# Prompt template for pattern extraction
pattern_extraction_prompt = """
You are an expert in AI design patterns.

An AI design pattern is a proven, reusable solution to a recurring problem 
in the design, development, or deployment of AI/ML systems. 
It captures the essence of a solution in a structured way, making it easier 
to apply in similar contexts. 

You will be given the text of a research paper. 
Your task is to carefully scan the text and extract **all AI design patterns** mentioned.

For each pattern, identify the following fields:
- Pattern Name
- Problem
- Context
- Solution
- Result
- Related Patterns
- Uses

AI Design Patterns are:
1. Classical AI
2. Generative AI Patterns
3. Agentic AI Patterns
4. Prompt Design Patterns
5. MLOps Patterns
6. AI–Human Interaction Patterns
7. LLM-specific Patterns
8. Tools Integration Patterns
9. Knowledge & Reasoning Patterns
10. Planning Patterns
11. Personalization Pattern

Return the output strictly as a JSON array.
Do not include explanations outside JSON.

Paper text:
{text}
"""

prompt = PromptTemplate(
    template=pattern_extraction_prompt,
    input_variables=["text"]
)

chain = prompt | llm


def load_text_file(file_path):
    with open(file_path, "r") as file:
        return file.read()

def remove_json_header_footer(text):
    start_index = text.find('[')
    end_index = text.rfind(']') + 1
    if start_index != -1 and end_index != -1:
        return text[start_index:end_index]
    return text

def extract_patterns_from_text(text):
    result = chain.invoke({"text": text})
    return remove_json_header_footer(result.content)

def extract_patterns(file_path):
    text = load_text_file(file_path)
    patterns = extract_patterns_from_text(text)
    return patterns

def save_patterns_to_file(patterns, output_path):
    with open(output_path, "w") as file:
        file.write(patterns)

if __name__ == "__main__":
    file_path = "cleaned_papers/cleaned_2307.07697v6.pdf.txt"
    patterns = extract_patterns(file_path)
    save_patterns_to_file(patterns, "extracted_patterns.json")
