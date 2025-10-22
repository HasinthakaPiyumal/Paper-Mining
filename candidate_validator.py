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

pattern_extraction_prompt = """
You are given a candidate AI design pattern. Your task is to evaluate it using 20 questions about problem clarity, solution robustness, context-awareness, results validation, and practical usability. Rate each question briefly as 0-10.

Output the result as a JSON array using these short keys:
Problem & Context (Questions 1-5):
    q1: A Recurring Problem – Does it state a specific, commonly encountered problem in software design that needs solving?
    q2: Is the problem non-trivial and likely to appear in various contexts?
    q3: A Specific Context – Does it describe the circumstances or conditions under which this problem typically arises?
    q4: Does the problem mention why naive or existing solutions are insufficient?
    q5: Is the pattern related to AI?

Solution Evaluation (Questions 6-10):
    q6: A Well-Defined Solution – Does it propose a concrete, structured solution involving specific roles, responsibilities, and collaborations between software elements (like classes or objects)?
    q7: Is the solution abstract enough to be applied in different ways, not just one specific implementation?
    q8: Does the solution encourage robustness against edge cases or unexpected inputs?
    q9: Does it discuss the benefits of applying this solution (e.g., improved flexibility, reusability, maintainability)?
    q10: Does it mention the potential drawbacks, limitations, or trade-offs involved?

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


- score calculate like this - as percentage of total rate of questions

Input Pattern Candidates:
{patterns_chunk}

Output Example:


 name:"NAME_OF_PATTERN,"q1": 8, "q2": 9, "q3": 7, "q4": 8, "q5": 10, "q6": 9, "q7": 8,"q8": 7, "q9": 9, "q10": 8, "q11": 9, "q12": 8, "q13": 7, "q14": 8, "q15": 9, "q16": 8, "q17": 9, "q18": 8, "q19": 9, "q20": 10,"score":85

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
    # file_path = "outputs/[25.10.21] - 02 - Added Retry Mechanism to Pattern Extraction/extracted_patterns.json"
    file_path = "outputs/[25.10.21] - 02 - Added Retry Mechanism to Pattern Extraction/interation_test/extracted_patterns-auto-reduced-1.json"
    validation_results = extract_patterns(file_path)
    save_patterns_to_file(validation_results, "/".join(file_path.split("/")[:-1]) + "/validated_patterns_extracted_patterns-auto-reduced-1.json")