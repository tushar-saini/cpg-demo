import argparse
from ingestion.llm_chunker import llm_chunking
from ingestion.chromadb import vectorize_chunks
import uvicorn

def start():
    uvicorn.run(
        "app:app",          # file:variable
        host="127.0.0.1",
        port=8000,
        reload=True         # auto-reload (dev only)
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest documents into ChromaDB")

    parser.add_argument(
        "--inpath",
        type=str,
        default='./data/data.json',
        required=True,
        help="Path to input JSON file"
    )

    parser.add_argument(
        "--outpath",
        type=str,
        required=True,
        default='./data/llm_chunks.json',
        help="Path to output JSON file"
    )

    # DO NOT CHANGE
    # parser.add_argument(
    #     "--chromadbpath",
    #     type=str,
    #     required=True,
    #     default='./chroma_db',
    #     help="Path to persisitent chroma db"
    # )

    args = parser.parse_args()

    llm_chunking(args.inpath, args.outpath)
    vectorize_chunks(args.outpath, './chroma_db')

    # After all the setup is done, we can start the server
    start()

