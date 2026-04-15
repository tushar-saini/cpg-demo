import argparse
import requests, json, re

from tqdm import tqdm


# Using locally hosted Ollama to generate chunks based on complete semantic meaning
OLLAMA_URL = "http://localhost:11434/api/generate"
# MODEL_NAME = "gemma4" 
MODEL_NAME = "qwen3:1.7b"

def generate_chunks(input_text):
    prompt = f"""
            You are an intelligent multi lingual text chunking system.

            Your task is to split the given paragraph into meaningful chunks such that:
            1. Each chunk preserves complete semantic meaning (no broken context).
            2. Do NOT split sentences in the middle.
            3. Group related sentences together based on topic or intent.
            4. Each chunk should be self-contained and understandable without needing adjacent chunks.
            5. Avoid chunks that are too small (single sentence unless necessary) or too large.
            6. Ensure important qualifiers (e.g., conditions, exceptions) stay in the same chunk.
            
            Output format:
            Return a JSON array of chunks like:
            [
              {{
                "chunk_id": 1,
                "sub_title": "<short descriptive title>",
                "content": "<chunk text>"
              }}
            ]
            
            Now process the following text:
            
            {input_text}
            """

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False
        }
    )

    result = response.json()
    return result.get("response", "Error generating response")

def clean_llm_output(text: str):
    text = text.strip()

    # Remove ```json or ```
    text = re.sub(r"^```json", "", text)
    text = re.sub(r"^```", "", text)
    text = re.sub(r"```$", "", text)

    return text.strip()

def llm_chunking_json(text: str):
    data = json.loads(clean_llm_output(generate_chunks(text)))
    return data

def llm_chunking(infile: str, outfile: str):
    all_chunks_data = []
    with open(infile, 'r') as f:
        data_json = json.load(f)


    for row in tqdm(data_json):
        llm_chunks = llm_chunking_json(row['body'])
        row['chunks'] = llm_chunks

    # Write back to the file
    with open(outfile, 'w') as file:
        json.dump(data_json, file, indent=4)


# Main function takes inpath and outpath
#    inpath: file to raw json data
#    outpath: path to save the output json file
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest documents into ChromaDB")

    parser.add_argument(
        "--inpath",
        type=str,
        required=True,
        help="Path to input JSON file"
    )

    parser.add_argument(
        "--outpath",
        type=str,
        required=True,
        help="Path to output JSON file"
    )

    args = parser.parse_args()

    llm_chunking(args.inpath, args.outpath)
