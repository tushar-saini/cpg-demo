import chromadb, json
from chromadb.utils import embedding_functions

def get_chroma_client(chromadb_path: str):
    client = chromadb.PersistentClient(path=chromadb_path)
    return client

# Create separate collections per country and language to enforce strict logical
# isolation of documents and prevent cross-tenant data leakage during retrieval.
def get_or_create_collection(client, country, language):
    collection_name = f"customer_content_{country}_{language}"

    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    return client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_fn
    )

def vectorize_chunks(chunk_file_json: str, chromadb_path: str):
    with open(chunk_file_json, 'r') as f:
        document_chunks = json.load(f)
    
    for doc_ in document_chunks:
    
        meta_data = {
                'content_id': doc_['content_id'],
                'country'   : doc_['country'],
                'language'  : doc_['language'],
                'type'      : doc_['type'],
                'version'   : doc_['version'],
                'title'     : doc_['title'],
                'updated_at': doc_['updated_at']
            }

        client = get_chroma_client(chromadb_path)
        
        collection = get_or_create_collection(
                client,
                doc_["country"],
                doc_["language"]
            )
        
        collection.add(
                documents=[c["content"] for c in doc_['chunks']],
                metadatas=[meta_data for c in doc_['chunks']],
                ids=[str(meta_data["content_id"]+str(c["chunk_id"])) for c in doc_['chunks']]
            )