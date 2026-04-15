import chromadb

DB_PATH = "./chroma_db"

def inspect():
    client = chromadb.PersistentClient(path=DB_PATH)

    collections = client.list_collections()

    print("\nCollections:")
    for c in collections:
        print("-", c.name)

    # Pick one collection
    collection = client.get_collection(collections[0].name)

    data = collection.get(limit=5)

    print("\nSample Records:\n")

    for i in range(len(data["ids"])):
        print("ID:", data["ids"][i])
        print("Document:", data["documents"][i])
        print("Metadata:", data["metadatas"][i])
        print("-" * 50)


if __name__ == "__main__":
    inspect()