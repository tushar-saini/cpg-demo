# retrieval/query.py
import chromadb

# TODO: If path is changed
DB_PATH = "./chroma_db"

def get_collection(country, language):
    client = chromadb.PersistentClient(path=DB_PATH)

    name = f"customer_content_{country}_{language}"

    return client.get_collection(name)


def query_docs(question, country, language, k=5):
    collection = get_collection(country, language)
    
    results = collection.query(
        query_texts=[question],
        n_results=k
    )
    

    docs = []

    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        docs.append({
            "text": doc,
            "metadata": meta,
            "distance": dist
        })

    return docs