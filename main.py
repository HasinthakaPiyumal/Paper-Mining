from pattern_extractor import extract_patterns, save_patterns_to_file
from paper_cleaner import clean_all_pdfs_in_folder

import os
import json

paper_folder = "papers"
output_folder = "cleaned_papers"
patterns_folder = "patterns" # If needed to save patterns separately
patterns_file = "extracted_patterns.json" # If needed to save all patterns in one file


def clean_papers():
    clean_all_pdfs_in_folder(paper_folder, output_folder)

def get_cleaned_file_list():
    cleaned_folder = output_folder
    return [os.path.join(cleaned_folder, f) for f in os.listdir(cleaned_folder) if f.endswith(".txt")]

def extract_patterns_from_all_files(file_list,auto_save=True,save_in_single_file=True,save_individual=True):
    """
    Extract patterns from a list of files.
    If auto_save is True, saves patterns to files.
    If save_individual is True, saves each file's patterns separately.
    If save_in_single_file is True, saves all patterns in a single file.
    If both are True, does both.
    """
    all_patterns = []
    for file_path in file_list:
        print(f"Processing file: {file_path}")
        patterns = extract_patterns(file_path)
        if auto_save:
            os.makedirs(patterns_folder, exist_ok=True)
            if save_individual:
                base_name = os.path.basename(file_path).replace("cleaned_", "").replace(".txt", "_patterns.json")
                output_path = os.path.join(patterns_folder, base_name)
                save_patterns_to_file(patterns, output_path)
                print(f"Saved individual patterns to: {output_path}")
            if save_in_single_file:
                json_patterns = json.loads(patterns)
                all_patterns.extend(json_patterns)
                with open(patterns_file, "w") as f:
                    json.dump(all_patterns, f, indent=2)
                print(f"Saved all patterns to: {patterns_file}")
    return all_patterns

if __name__ == "__main__":
    clean_papers() # Uncomment if cleaning is needed
    cleaned_files = get_cleaned_file_list()
    extract_patterns_from_all_files(cleaned_files)