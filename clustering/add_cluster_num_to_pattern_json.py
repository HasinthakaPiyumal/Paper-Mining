import json,pandas as pd

def add_cluster_numbers_to_json(input_file, output_file, cluster_file):

    patterns_with_clusters = pd.read_json(cluster_file)

    with open(input_file, 'r') as f:
        data = json.load(f)

    for idx, item in enumerate(data):
        pattern_name = item.get("Pattern Name")
        if pattern_name == patterns_with_clusters.loc[idx]["Pattern Name"]:
            item["Cluster Number"] = int(patterns_with_clusters.loc[idx]["cluster"])
        else:
            item["Cluster Number"] = None
        
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)

if __name__ == "__main__":
    folder_path = "/home/hasinthaka/Documents/Projects/AI/AI Pattern Mining/Paper Mining/outputs/[25.09.12] - 02 - Prompt, Temperature, Embedding Model, Clustering changes"
    input_file = f"{folder_path}/extracted_patterns.json"
    output_file = f"{folder_path}/full_patterns_with_clusters.json"
    cluster_file = f"{folder_path}/patterns_with_26_clusters.json" # Output from clustering notebook
    add_cluster_numbers_to_json(input_file, output_file, cluster_file)
    df = pd.read_json(output_file)
    df.to_csv("full_patterns_with_clusters.csv", index=False)