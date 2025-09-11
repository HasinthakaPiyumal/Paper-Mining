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
# pattern_extraction_prompt = """
# You are an expert in AI design patterns.

# An AI design pattern is a proven, reusable solution to a recurring problem 
# in the design, development, or deployment of AI/ML systems. 
# It captures the essence of a solution in a structured way, making it easier 
# to apply in similar contexts. 

# You will be given the text of a research paper. 
# Your task is to carefully scan the text and extract **all AI design patterns** mentioned.

# For each pattern, identify the following fields:
# - Pattern Name
# - Problem
# - Context
# - Solution
# - Result
# - Related Patterns
# - Uses

# AI Design Patterns are:
# 1. Classical AI
# 2. Generative AI Patterns
# 3. Agentic AI Patterns
# 4. Prompt Design Patterns
# 5. MLOps Patterns
# 6. AI–Human Interaction Patterns
# 7. LLM-specific Patterns
# 8. Tools Integration Patterns
# 9. Knowledge & Reasoning Patterns
# 10. Planning Patterns
# 11. Personalization Pattern

# Return the output strictly as a JSON array.
# Do not include explanations outside JSON.

# Paper text:
# {text}
# """

pattern_extraction_prompt = """
An AI design pattern is a proven, reusable solution to a recurring problem specifically within AI/ML system design, development, or deployment. It addresses challenges inherent to building machine learning, agentic behavior, or data-driven intelligence.

**Exclusion Criteria:**
Do NOT extract general software engineering, data architecture, or DevOps patterns. These are often used to build AI systems but are not AI design patterns themselves. Specifically, ignore concepts like:
- **General Architectural Patterns:** Microservice Architecture, Lambda/Kappa Architecture, Gateway Routing, Layered Architecture (MultiLayer Pattern).
- **Classic Software Design Patterns:** Strategy Pattern, Separation of Concerns, Facade/Adapter patterns (e.g., "Wrap BlackBox Packages").
- **General Data Engineering Patterns:** Data Lake, Batch Serving, generic Workflow Pipelines.
- **General DevOps/Process Patterns:** Continuous Integration and Deployment (CI/CD), general testing principles, code reuse, and versioning strategies.

Given a research paper text, extract only the true AI design patterns mentioned.

For each pattern, include:
- Pattern Name
- Problem
- Context
- Solution
- Result
- Related Patterns (only other extracted patterns)
- Uses

If patterns are mostly similar in their problem, solution, or context, merge them into a single entry. When merging, combine their names, uses, and related patterns.

Recognized AI Design Pattern Categories:
1. Classical AI
2. Generative AI
3. Agentic AI
4. Prompt Design
5. MLOps (only if specific to ML workflows, not general deployment)
6. AI–Human Interaction
7. LLM-specific
8. Tools Integration
9. Knowledge & Reasoning
10. Planning
11. Personalization

Examples:

Valid AI Pattern:
"Pattern Name": "Hierarchical Planning",
"Problem": "Complex, long-horizon tasks for embodied agents are difficult to plan directly.",
"Context": "Embodied agents following multi-step instructions.",
"Solution": "Decompose planning into high-level and low-level planners.",
"Result": "Improves planning efficiency for long-horizon tasks.",
"Related Patterns": "LLM as a Planner, Grounded Replanning",
"Uses": "Robotics, Vision-and-language navigation"

Pattern to IGNORE (Not an AI Pattern):
"Pattern Name": "Microservice Architecture",
"Problem": "ML applications may be confined to some known ML frameworks.",
"Solution": "Enable data scientists to make ML frameworks available through microservices."

Return the output strictly as a JSON array.

Paper text:
{text}
"""


summary_prompt = """
You are an expert in AI design patterns. 
Your task is to combine the following AI design patterns into a single, unified pattern. 
Use information from all patterns to produce one coherent pattern that includes:

- Pattern Name
- Problem
- Context
- Solution
- Result
- Related Patterns
- Uses

Return strictly as JSON. Do not add extra text, explanations, or formatting.

Patterns to combine:
{patterns_text}
"""

def load_text_file(file_path):
    with open(file_path, "r") as file:
        return file.read()

def remove_json_header_footer(text):
    start_index = text.find('[')
    end_index = text.rfind(']') + 1
    if start_index != -1 and end_index != -1:
        return text[start_index:end_index]
    return text

def remove_json_annotations(text):
    text = text.replace("```json", "").replace("```", "")
    return text

def extract_patterns_from_text(text):

    prompt = PromptTemplate(
        template=pattern_extraction_prompt,
        input_variables=["text"]
    )

    chain = prompt | llm
    result = chain.invoke({"text": text})
    return remove_json_header_footer(result.content)

def extract_patterns(file_path):
    text = load_text_file(file_path)
    patterns = extract_patterns_from_text(text)
    return patterns

def save_patterns_to_file(patterns, output_path):
    with open(output_path, "w") as file:
        file.write(patterns)

def summarize_patterns(patterns):
    prompt = PromptTemplate(
        template=summary_prompt,
        input_variables=["patterns_text"]
    )

    chain = prompt | llm
    summary = chain.invoke({"patterns_text": patterns})
    return remove_json_annotations(summary.content)

if __name__ == "__main__":
    file_path = "cleaned_papers/cleaned_Song_LLM-Planner_Few-Shot_Grounded_Planning_for_Embodied_Agents_with_Large_Language_ICCV_2023_paper.pdf.txt"
    print('Extracting patterns from:', file_path)
    patterns = extract_patterns(file_path)
    print('Extracted')
    save_patterns_to_file(patterns, "test_pattern_extraction.json")
