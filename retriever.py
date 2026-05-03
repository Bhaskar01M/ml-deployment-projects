import json
import math
from pathlib import Path

class Retriever:
    def __init__(self, embeddings_file="embeddings.json"):
        self.embeddings = []
        self.load_embeddings(embeddings_file)
    
    def load_embeddings(self, filename):
        """Load embeddings from JSON file"""
        try:
            with open(filename, 'r') as f:
                self.embeddings = json.load(f)
            print(f"✅ Loaded {len(self.embeddings)} embeddings from {filename}\n")
        except FileNotFoundError:
            print(f"❌ File not found: {filename}")
            print("   Run demo_embeddings_local.py first!")
            return False
        except Exception as e:
            print(f"❌ Error loading embeddings: {e}")
            return False
    
    def cosine_similarity(self, vec1, vec2):
        """Calculate cosine similarity between two vectors"""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a ** 2 for a in vec1))
        norm2 = math.sqrt(sum(b ** 2 for b in vec2))
        
        if norm1 == 0 or norm2 == 0:
            return 0
        
        return dot_product / (norm1 * norm2)
    
    def retrieve(self, query_embedding, k=5):
        """Find top-k most similar chunks to the query"""
        scores = []
        
        for i, embedding_obj in enumerate(self.embeddings):
            similarity = self.cosine_similarity(
                query_embedding,
                embedding_obj["embedding"]
            )
            scores.append({
                "chunk_id": i,
                "text": embedding_obj["text"],
                "similarity": similarity
            })
        
        # Sort by similarity (highest first)
        scores.sort(key=lambda x: x["similarity"], reverse=True)
        
        # Return top-k
        return scores[:k]


if __name__ == "__main__":
    print("="*60)
    print("RETRIEVER TEST")
    print("="*60 + "\n")
    
    # Initialize retriever
    retriever = Retriever()
    
    # Create a random query embedding for testing
    import random
    query_embedding = [random.uniform(-1, 1) for _ in range(1536)]
    
    print("🔍 Searching for similar chunks...\n")
    
    # Retrieve top 5
    results = retriever.retrieve(query_embedding, k=5)
    
    print("📌 TOP 5 MOST SIMILAR CHUNKS:\n")
    for i, result in enumerate(results, 1):
        print(f"{i}. Similarity: {result['similarity']:.4f}")
        print(f"   Text: {result['text']}")
        print()
    
    print("="*60)
    print("✅ RETRIEVER WORKING!")
    print("="*60)