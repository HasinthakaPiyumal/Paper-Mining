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
    input_file = "../extracted_patterns.json"
    output_file = f"full_patterns_with_clusters.json"
    cluster_file = "patterns_with_50_clusters.json" # Output from clustering notebook
    add_cluster_numbers_to_json(input_file, output_file, cluster_file)