import json
import random
from pathlib import Path
from pdf_processor import PDFProcessor

class DemoEmbeddingsGenerator:
    """Demo version - creates fake embeddings locally without API calls"""
    
    def __init__(self):
        self.embeddings = []
    
    def generate_demo_embeddings(self, chunks):
        """Generate FAKE embeddings locally for demo purposes"""
        print(f"🔄 Generating DEMO embeddings for {len(chunks)} chunks...")
        print("   (These are fake embeddings for testing - no API calls)\n")
        
        for i, chunk in enumerate(chunks):
            # Create a fake 1536-dimensional embedding (OpenAI's size)
            fake_embedding = [random.uniform(-1, 1) for _ in range(1536)]
            
            self.embeddings.append({
                "chunk_id": i,
                "text": chunk[:100] + "...",
                "embedding": fake_embedding
            })
            
            if (i + 1) % 100 == 0:
                print(f"   ✅ Generated {i + 1}/{len(chunks)} embeddings")
        
        print(f"\n✅ Generated {len(self.embeddings)} DEMO embeddings\n")
        return self.embeddings
    
    def save_embeddings(self, filename="embeddings.json"):
        """Save embeddings to JSON file"""
        try:
            with open(filename, 'w') as f:
                json.dump(self.embeddings, f)
            print(f"💾 Saved embeddings to {filename}")
            return True
        except Exception as e:
            print(f"❌ Error saving: {e}")
            return False


if __name__ == "__main__":
    print("="*60)
    print("DEMO EMBEDDINGS GENERATOR (No API calls)")
    print("="*60 + "\n")
    
    processor = PDFProcessor()
    
    if processor.load_pdfs():
        chunks = processor.get_all_chunks()
        print(f"✅ Got {len(chunks)} chunks\n")
        
        generator = DemoEmbeddingsGenerator()
        generator.generate_demo_embeddings(chunks)
        generator.save_embeddings()
        
        print("="*60)
        print("✅ DEMO EMBEDDINGS COMPLETE!")
        print("="*60)
    else:
        print("Failed to load PDFs")