from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os

def get_embedding_model():
    return GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

def generate_embeddings(texts):
    embedding_model = get_embedding_model()
    embeddings = embedding_model.embed_documents(texts)
    return embeddings

def pattern_combiner(patterns):
    combined_patterns = []

    for pattern in patterns:
        pattern_text = f"Pattern Name: {pattern.get('Pattern Name', 'Unnamed Pattern')}\n"
        pattern_text += f"Problem: {pattern.get('Problem', '')}\n"
        pattern_text += f"Context: {pattern.get('Context', '')}\n"
        pattern_text += f"Solution: {pattern.get('Solution', '')}\n"
        pattern_text += f"Result: {pattern.get('Result', '')}\n"
        pattern_text += f"Uses: {', '.join(pattern.get('Uses', []))}\n"
        combined_patterns.append(pattern_text)
    return combined_patterns

def add_embeddings(embeddings,names, file_path):
    import json,pandas as pd,os
    if os.path.exists(file_path):   
        df1 = pd.read_csv(file_path)
        df2 = pd.DataFrame(embeddings)
        df2['Pattern Name'] = names
        df = pd.concat([df1, df2], ignore_index=True)
        df.to_csv(file_path, index=False)
    else:
        df = pd.DataFrame(embeddings)
        df['Pattern Name'] = names
        df.to_csv(file_path, index=False)

def main(base_output):
    import json,time,os
    with open(f'{base_output}/extracted_patterns.json', 'r') as f:
        patterns = json.load(f)
    skip_count = 40
    all_embeddings = []
    for i in range((len(patterns)//skip_count)+1):
        start_index = i * skip_count
        end_index = start_index + skip_count
        end_index = min(end_index, len(patterns))
        combined_patterns = pattern_combiner(patterns[start_index:end_index])
        print(f'Generating embeddings for patterns {start_index} to {end_index}...')
        embeddings = generate_embeddings(combined_patterns)
        os.remove(f'{base_output}/pattern_embeddings_inprogress.csv') if os.path.exists(f'{base_output}/pattern_embeddings_inprogress.csv') else None
        print(f'Adding embeddings for patterns {start_index} to {end_index}...')
        print(f'Length of embeddings: {len(embeddings)}')
        add_embeddings(embeddings,[pattern.get('Pattern Name', 'Unnamed Pattern') for pattern in patterns[start_index:end_index]], f'{base_output}/pattern_embeddings_inprogress.csv')
        all_embeddings.extend(embeddings)
        if end_index != len(patterns):
            for _ in range(60):
                print(f'Waiting... {60-_} seconds remaining.', end='\r')
                time.sleep(1)
    add_embeddings(all_embeddings,[pattern.get('Pattern Name', 'Unnamed Pattern') for pattern in patterns], f'{base_output}/pattern_embeddings.csv')
    print(f'Added embeddings for all patterns.')

if __name__ == "__main__":
    base_output = "outputs/[25.09.12] - 02 - Prompt, Temperature, Embedding Model, Clustering changes/"
    main(base_output)
