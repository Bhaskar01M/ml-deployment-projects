import PyPDF2
from pathlib import Path

class PDFProcessor:
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.documents = []
    
    def load_pdfs(self):
        """Load all PDF files from the data directory."""
        pdf_files = list(self.data_dir.glob("*.pdf"))
        
        if not pdf_files:
            print("❌ No PDF files found in data/ folder")
            return False
        
        print(f"📄 Found {len(pdf_files)} PDF files\n")
        
        for pdf_file in pdf_files:
            print(f"📖 Loading: {pdf_file.name}")
            text = self.extract_text(pdf_file)
            self.documents.append({
                "filename": pdf_file.name,
                "text": text,
                "length": len(text)
            })
            print(f"   ✅ Extracted {len(text)} characters\n")
        
        return True
    
    def extract_text(self, pdf_path):
        """Extract text from a PDF file."""
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                num_pages = len(pdf_reader.pages)
                
                for page_num in range(num_pages):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text()
            
            return text
        except Exception as e:
            print(f"   ❌ Error reading {pdf_path}: {e}")
            return ""
    
    def chunk_text(self, text, chunk_size=500, overlap=100):
        """Split text into overlapping chunks."""
        chunks = []
        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i:i + chunk_size]
            if len(chunk.strip()) > 0:
                chunks.append(chunk)
        return chunks
    
    def get_all_chunks(self):
        """Get all chunks from all documents."""
        all_chunks = []
        
        for doc in self.documents:
            chunks = self.chunk_text(doc["text"])
            all_chunks.extend(chunks)
        
        return all_chunks
    
    def print_summary(self):
        """Print a summary of loaded documents."""
        print("\n" + "="*60)
        print("📊 DOCUMENT SUMMARY")
        print("="*60)
        
        total_chars = 0
        for doc in self.documents:
            print(f"\n📄 {doc['filename']}")
            print(f"   Characters: {doc['length']:,}")
            total_chars += doc['length']
        
        all_chunks = self.get_all_chunks()
        print(f"\n✅ Total Documents: {len(self.documents)}")
        print(f"✅ Total Characters: {total_chars:,}")
        print(f"✅ Total Chunks: {len(all_chunks)}")
        print("="*60 + "\n")


if __name__ == "__main__":
    # Create processor and load PDFs
    processor = PDFProcessor()
    
    if processor.load_pdfs():
        processor.print_summary()
        
        # Show first chunk as example
        chunks = processor.get_all_chunks()
        if chunks:
            print("📌 FIRST CHUNK (example):")
            print("-" * 60)
            print(chunks[0][:300] + "...")
            print("-" * 60)
    else:
        print("Failed to load PDFs")