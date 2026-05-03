import json
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from pdf_processor import PDFProcessor

# Load .env file
load_dotenv()

class EmbeddingsGenerator:
    def __init__(self, model="text-embedding-3-small"):
        self.client = OpenAI()
        self.model = model
        self.embeddings = []
        
    def generate_embeddings(self, chunks):
        """Generate embeddings for text chunks using OpenAI."""
        print(f"🔄 Generating embeddings for {len(chunks)} chunks...")
        print(f"   Using model: {self.model}\n")
        
        for i, chunk in enumerate(chunks):
            try:
                # Call OpenAI API to get embedding
                response = self.client.embeddings.create(
                    input=chunk,
                    model=self.model
                )
                
                embedding = response.data[0].embedding
                
                self.embeddings.append({
                    "chunk_id": i,
                    "text": chunk[:100] + "...",  # Store first 100 chars
                    "embedding": embedding
                })
                
                # Print progress
                if (i + 1) % 50 == 0:
                    print(f"   ✅ Generated {i + 1}/{len(chunks)} embeddings")
            
            except Exception as e:
                print(f"   ❌ Error generating embedding for chunk {i}: {e}")
        
        print(f"\n✅ Generated {len(self.embeddings)} embeddings\n")
        return self.embeddings
    
    def save_embeddings(self, filename="embeddings.json"):
        """Save embeddings to a JSON file."""
        try:
            with open(filename, 'w') as f:
                json.dump(self.embeddings, f)
            print(f"💾 Saved embeddings to {filename}")
            return True
        except Exception as e:
            print(f"❌ Error saving embeddings: {e}")
            return False


if __name__ == "__main__":
    # Load PDFs and get chunks
    print("="*60)
    print("EMBEDDINGS GENERATOR")
    print("="*60 + "\n")
    
    processor = PDFProcessor()
    
    if processor.load_pdfs():
        chunks = processor.get_all_chunks()
        print(f"✅ Got {len(chunks)} chunks\n")
        
        # Generate embeddings
        generator = EmbeddingsGenerator()
        generator.generate_embeddings(chunks)
        
        # Save embeddings
        generator.save_embeddings()
        
        print("="*60)
        print("✅ EMBEDDINGS GENERATION COMPLETE!")
        print("="*60)
    else:
        print("Failed to load PDFs")