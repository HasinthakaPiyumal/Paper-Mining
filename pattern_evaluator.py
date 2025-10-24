import os,json,time
import getpass
from langchain.chat_models import init_chat_model

if not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for Google Gemini: ")

llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai", temperature=0)

evaluation_prompt = """
You are an expert in **AI Design Pattern Evaluation**.
Your task is to compare both lists and evaluate how accurately the generated patterns match the correct ones in **meaning and structure**.

Compare two lists of AI design patterns:

1. CorrectPatterns – verified ground truth.
2. GeneratedPatterns – AI-generated list.

Match patterns **semantically** by Name, Problem, Context, and Solution (not exact text).  
Minor wording changes are fine if the meaning is the same.

Return only this JSON:

  "precision": float,
  "recall": float,
  "f1": float,
  "correct": ["patterns correctly matched between both lists"],
  "missing": ["patterns in CorrectPatterns but not in GeneratedPatterns"],
  "extra": ["patterns in GeneratedPatterns but not in CorrectPatterns"]


Rules:
- precision = correct_matches / total_generated  
- recall = correct_matches / total_correct  
- f1 = 2 * (precision * recall) / (precision + recall)  
- Round numbers to 3 decimals.  
- Output valid JSON only.

Data:
CorrectPatterns:
{correct_list}

GeneratedPatterns:
{generated_json}
"""

def remove_json_header_footer(text):
    start_index = text.find('{')
    end_index = text.rfind('}') + 1
    if start_index != -1 and end_index != -1:
        return text[start_index:end_index]
    return text

def generate_evaluation(correct_list, generated_json):
    prompt_filled = evaluation_prompt.format(
        correct_list=correct_list,
        generated_json=generated_json
    )
    response = llm.invoke(prompt_filled)
    try:
        result = json.loads(remove_json_header_footer(response.content))
    except json.JSONDecodeError:
        # Retry logic could be implemented here
        result = {"error": "Failed to parse JSON from LLM response."}
    return result

def load_data_and_evaluate(file_path_correct, file_path_generated):
    correct_patterns = open(file_path_correct).read()
    generated_patterns = open(file_path_generated).read()
    evaluation_result = generate_evaluation(correct_patterns, generated_patterns)
    return evaluation_result

def evaluate_patterns(truth_file, gen_file):
    print(f"Evaluating patterns from ({gen_file.split('/')[-1][:50]}) against ground truth ({truth_file.split('/')[-1][:50]})...")
    result_file = f"{os.path.dirname(truth_file)}/_eval_result_{gen_file.split('/')[-1][:50]}++{truth_file.split('/')[-1][:50]}.json"
    evaluation_result = load_data_and_evaluate(truth_file, gen_file)
    with open(result_file, "w") as f:
        json.dump(evaluation_result, f, indent=2)
    print(f"Evaluation results saved to {result_file}","\n","=="*50,"\n")

if __name__ == "__main__":  
    truth_file = "outputs/[25.10.23] - 03 - Optimized prompts/ground_truth_list/Agent design pattern catalogue: A collection of architectural patterns for foundation model based agents.txt"
    gen_file = "outputs/[25.10.23] - 03 - Optimized prompts/patterns/Agent design pattern catalogue_ A collection of architectural patterns for foundation model based agents.pdf_patterns.json"
    evaluate_patterns(truth_file, gen_file)

    truth_file = "outputs/[25.10.23] - 03 - Optimized prompts/ground_truth_list/Washizaki, H.; et al.. Software-Engineering Design Patterns for Machine Learning Applications. Computer 55(3) 2022. .txt"
    gen_file = "outputs/[25.10.23] - 03 - Optimized prompts/patterns/Software-Engineering_Design_Patterns_for_Machine_Learning_Applications.pdf_patterns.json"
    evaluate_patterns(truth_file, gen_file)

    truth_file = "outputs/[25.10.23] - 03 - Optimized prompts/ground_truth_list/Washizaki, H.; et al.. Software-Engineering Design Patterns for Machine Learning Applications. Computer 55(3) 2022. .txt"
    gen_file = "outputs/[25.10.21] - 02 - Added Retry Mechanism to Pattern Extraction/mannual_tests/Software Engineering Design Patterns for Machine Learning Applications_0.json"
    evaluate_patterns(truth_file, gen_file)

    truth_file = "outputs/[25.10.23] - 03 - Optimized prompts/ground_truth_list/Github Gen AI patterns.txt"
    gen_file = "outputs/[25.10.23] - 03 - Optimized prompts/patterns/gen_ai_patterns_github_patterns.json"
    evaluate_patterns(truth_file, gen_file)
