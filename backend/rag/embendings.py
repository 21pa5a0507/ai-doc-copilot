from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')


def get_embending(text):
    # SentenceTransformer uses encode() for embeddings
    emb = model.encode(text)

    # return plain Python list for compatibility with FAISS vector handling
    return emb.tolist() if hasattr(emb, 'tolist') else list(emb)
