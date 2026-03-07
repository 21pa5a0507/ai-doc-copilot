import faiss
import numpy as np

class VectorStore:
    def __init__(self, dimension=3072):
        self.dimension = dimension
        
        self.index = faiss.IndexFlatL2(dimension)

        self.text_chunks = []
    
    def add(self, embedding, text):

        vector = np.array[embedding].astype("float32")

        self.index.add(vector)

        self.text_chunks.append(text)
    
    def search(self, embedding, k=3):

        vector = np.array[embedding].astype("float32")

        distances, indices = self.search(vector, k)

        results = []

        for i in indices[0]:
            results.append(self.text_chunks[i])
        
        return results