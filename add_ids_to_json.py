import json

def add_ids_to_json(input_file, output_file):
    with open(input_file, 'r') as f:
        data = json.load(f)

    for idx, item in enumerate(data):
        item['id'] = idx + 1

    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)

if __name__ == "__main__":
    add_ids_to_json('patterns_embeddings.json', 'patterns_embeddings_with_ids.json')