import os,json,time,threading
from pathlib import Path
import getpass
from langchain.chat_models import init_chat_model
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence

if not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for Google Gemini: ")

llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai", temperature=0)


"""
Prompt template for validating AI design pattern candidates using a comprehensive 20-question framework.

This prompt evaluates candidate patterns across five key dimensions to ensure they meet the quality
standards of well-defined design patterns. Each question is answered as "Yes", "No", or "Partially",
with a scoring system (Yes=5, Partially=2.5, No=0) for a maximum score of 100.

Evaluation Framework:

Problem & Context (Questions 1-5):
    q1: Does the problem clearly describe a recurring AI challenge?
    q2: Is the problem specific enough to understand what is failing or suboptimal?
    q3: Does the problem mention why naive or existing solutions are insufficient?
    q4: Is the context explained clearly, showing where and when the problem occurs?
    q5: Does the pattern highlight constraints or dynamic aspects of the context?

Solution Evaluation (Questions 6-10):
    q6: Is the solution actionable and reproducible for someone implementing it?
    q7: Does the solution show how it directly addresses the problem?
    q8: Is the solution generalizable beyond a single dataset, tool, or model?
    q9: Does the solution encourage robustness against edge cases or unexpected inputs?
    q10: Are any trade-offs or limitations of the solution mentioned?

Result & Validation (Questions 11-14):
    q11: Does the pattern describe measurable improvement or benefits?
    q12: Are the results clearly linked to the problem and solution?
    q13: Does it provide evidence (quantitative or qualitative) that the solution works?
    q14: Are the results likely to be reproducible in other settings?

Related Patterns & Reusability (Questions 15-18):
    q15: Does the pattern relate to other known patterns in a meaningful way?
    q16: Can the pattern be combined with other patterns to solve larger problems?
    q17: Is the pattern reusable for similar AI tasks in different domains?
    q18: Does the pattern indicate when it should or should not be used?

Practical Insight (Questions 19-20):
    q19: Does the pattern provide insight that could help engineers avoid common mistakes?
    q20: Does it encourage learning and improvement in AI system design beyond just solving 
         the immediate problem?

Returns:
    JSON array with pattern name, answers for q1-q20, and calculated score.
"""

pattern_extraction_prompt = """
You are given a candidate AI design pattern. Your task is to evaluate it using 20 questions about problem clarity, solution robustness, context-awareness, results validation, and practical usability. Answer each question briefly as "Yes", "No", or "Partially".

Output the result as a JSON array using these short keys:

Problem & Context (Questions 1-5):
    q1: Does the problem clearly describe a recurring AI challenge?
    q2: Is the problem specific enough to understand what is failing or suboptimal?
    q3: Does the problem mention why naive or existing solutions are insufficient?
    q4: Is the context explained clearly, showing where and when the problem occurs?
    q5: Is it related to AI?

Solution Evaluation (Questions 6-10):
    q6: Is the solution actionable and reproducible for someone implementing it?
    q7: Does the solution show how it directly addresses the problem?
    q8: Is the solution generalizable beyond a single dataset, tool, or model?
    q9: Does the solution encourage robustness against edge cases or unexpected inputs?
    q10: Are any trade-offs or limitations of the solution mentioned?

Result & Validation (Questions 11-14):
    q11: Does the pattern describe measurable improvement or benefits?
    q12: Are the results clearly linked to the problem and solution?
    q13: Does it provide evidence (quantitative or qualitative) that the solution works?
    q14: Are the results likely to be reproducible in other settings?

Related Patterns & Reusability (Questions 15-18):
    q15: Does the pattern relate to other known patterns in a meaningful way?
    q16: Can the pattern be combined with other patterns to solve larger problems?
    q17: Is the pattern reusable for similar AI tasks in different domains?
    q18: Does the pattern indicate when it should or should not be used?

Practical Insight (Questions 19-20):
    q19: Does the pattern provide insight that could help engineers avoid common mistakes?
    q20: Does it encourage learning and improvement in AI system design beyond just solving 
         the immediate problem?


- score calculate like this - yes 5,no 0 , partialy 2.5 - (max total 100)

Input Pattern Candidates:
{patterns_chunk}

Output Example:


 "name":NAME_OF_PATTERN,"q1": "Yes", "q2": "Yes", "q3": "Partially", "q4": "Yes", "q5": "Yes", "q6": "Yes", "q7": "Yes","q8": "Yes", "q9": "Yes", "q10": "Partially", "q11": "Yes", "q12": "Yes", "q13": "Yes", "q14": "Yes", "q15": "Yes", "q16": "Yes", "q17": "Yes", "q18": "Yes", "q19": "Yes", "q20": "Yes","score":100

"""

retry_prompt = """
if there is any issue with bellow json format, correct it and return only the json array. if not return the same json.

Extracted patterns so far:
{extracted_patterns}
"""
iter_c = 0

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

def validate_candidate_patterns(extracted_patterns):
    global iter_c
    print(f"Validation iteration: {iter_c+1}")
    
    # Timer variables
    start_time = time.time()
    timer_running = True
    
    def print_timer():
        while timer_running:
            elapsed = time.time() - start_time
            print(f"\r     -Elapsed time: {elapsed:.1f}s", end='', flush=True)
            time.sleep(1)
    
    # Start timer thread
    timer_thread = threading.Thread(target=print_timer, daemon=True)
    timer_thread.start()
    
    try:
        prompt = pattern_extraction_prompt.format(patterns_chunk=extracted_patterns)
        open('debug_prompt.txt', 'w').write(prompt)
        result = llm.invoke(prompt)
        json.loads(remove_json_header_footer(result.content))

    except Exception as e:
        print(f"Error during validation: {e}")
        print(f"\n     -Intermediate validation result:\n{remove_json_header_footer(result.content)}\n")
        prompt_2 = retry_prompt.format(extracted_patterns=remove_json_header_footer(result.content))
        result = llm.invoke(prompt_2)
    finally:
        timer_running = False
        time.sleep(0.1)  # Give timer thread time to finish
        elapsed_total = time.time() - start_time
        print(f"\r     -Total time: {elapsed_total:.2f}s\n")

    iter_c += 1
    return remove_json_header_footer(result.content)

def extract_patterns(file_path):
    text = load_text_file(file_path)
    json_patterns = json.loads(text)
    patterns_chunks = []
    validation_results = []
    chunk_size = 10
    for i in range(0, len(json_patterns), chunk_size):
        chunk = json_patterns[i:i + chunk_size]
        chunk_text = json.dumps(chunk, indent=2)
        patterns_chunks.append(chunk_text)
    
    for chunk in patterns_chunks[:11]:
        validation_result = validate_candidate_patterns(chunk)
        validation_results.extend(json.loads(validation_result))
    
    return validation_results

def save_patterns_to_file(patterns, output_path):
    print(f"Saving validated patterns to: {output_path}")
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    with open(output_path, "w") as file:
        json.dump(patterns, file, indent=2)

if __name__ == "__main__":
    file_path = "outputs/[25.10.21] - 02 - Added Retry Mechanism to Pattern Extraction/extracted_patterns.json"
    validation_results = extract_patterns(file_path)
    save_patterns_to_file(validation_results, "/".join(file_path.split("/")[:-1]) + "/validated_patterns_1.json")