import json
def group_by_key(items, key):
    grouped = {}
    for item in items:
        key_value = item.get(key)
        if key_value not in grouped:
            grouped[key_value] = []
        grouped[key_value].append(item)
    return grouped

def open_json_file_as_object(file_path):
    with open(file_path, "r") as file:
        return json.load(file)

if __name__ == "__main__":
    input_file = "clustering/full_patterns_with_clusters.json"
    output_file = "summarized_patterns.json"

    print(f"Loading patterns from {input_file}")
    patterns = open_json_file_as_object(input_file)
    print(f"Loaded {len(patterns)} patterns")
    print("Grouping patterns by cluster number")
    patterns_by_cluster = group_by_key(patterns, "Cluster Number")
    print(f"Found {len(patterns_by_cluster)} clusters")

    summarized_patterns = []
    for cluster_num, cluster_patterns in patterns_by_cluster.items():
        if cluster_num is not None:
            from pattern_extractor import summarize_patterns
            patterns_text = "\n".join([json.dumps(p) for p in cluster_patterns])
            print(f"Summarizing cluster {cluster_num} with {len(cluster_patterns)} patterns")
            # print(cluster_patterns)
            summary = summarize_patterns(patterns_text)
            print("Summary:")
            # print()
            # print()
            # print()
            # print()
            print(summary)
            try:
                summary_json = json.loads(summary)
                if isinstance(summary_json, list):
                    summarized_patterns.extend(summary_json)
                else:
                    summarized_patterns.append(summary_json)
                print(f"Cluster {cluster_num} summarized successfully")
            except json.JSONDecodeError:
                print(f"Failed to parse summary for cluster {cluster_num}: {summary}")


    with open(output_file, "w") as f:
        json.dump(summarized_patterns, f, indent=2)

    print(f"Summarized patterns saved to {output_file}")