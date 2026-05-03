from dotenv import load_dotenv
from openai import OpenAI
from retriever import Retriever
import random

# Load environment
load_dotenv()

class RAGChatbot:
    def __init__(self, embeddings_file="embeddings.json"):
        self.client = OpenAI()
        self.retriever = Retriever(embeddings_file)
        self.model = "gpt-3.5-turbo"
    
    def generate_query_embedding(self, query):
        """
        Generate embedding for user query.
        For demo: create a fake embedding (later use OpenAI API)
        """
        # DEMO: Create fake embedding
        return [random.uniform(-1, 1) for _ in range(1536)]
    
    def answer_question(self, question, k=5):
        """Answer a question using RAG"""
        print(f"\n🤔 Question: {question}\n")
        
        # Step 1: Generate embedding for query
        print("📍 Step 1: Embedding query...")
        query_embedding = self.generate_query_embedding(question)
        print("   ✅ Query embedded\n")
        
        # Step 2: Retrieve relevant chunks
        print(f"📍 Step 2: Retrieving top {k} chunks...")
        chunks = self.retriever.retrieve(query_embedding, k=k)
        print(f"   ✅ Retrieved {len(chunks)} chunks\n")
        
        # Step 3: Create context from chunks
        print("📍 Step 3: Building context...")
        context = "\n---\n".join([chunk["text"] for chunk in chunks])
        print("   ✅ Context built\n")
        
        # Step 4: Send to LLM
        print("📍 Step 4: Generating answer with ChatGPT...")
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant. Answer questions based on the provided context. If the context doesn't contain the answer, say 'I don't have enough information to answer this.'"
                    },
                    {
                        "role": "user",
                        "content": f"Context:\n{context}\n\nQuestion: {question}"
                    }
                ],
                temperature=0.7,
                max_tokens=300
            )
            
            answer = response.choices[0].message.content
            print("   ✅ Answer generated\n")
            
        except Exception as e:
            answer = f"❌ Error: {str(e)}\n(This might be a billing issue on your OpenAI account)"
        
        return answer
    
    def chat(self):
        """Interactive chat mode"""
        print("\n" + "="*60)
        print("🤖 RAG CHATBOT - Interactive Mode")
        print("="*60)
        print("Type your questions below. Type 'exit' to quit.\n")
        
        while True:
            question = input("You: ").strip()
            
            if question.lower() == 'exit':
                print("\n👋 Goodbye!")
                break
            
            if not question:
                continue
            
            answer = self.answer_question(question)
            print(f"\n🤖 Assistant: {answer}")
            print("\n" + "-"*60 + "\n")


if __name__ == "__main__":
    chatbot = RAGChatbot()
    
    # Test with a few questions
    print("="*60)
    print("RAG CHATBOT - DEMO")
    print("="*60)
    
    test_questions = [
        "What is the Transformer architecture?",
        "How does BERT work?",
        "What is attention mechanism?"
    ]
    
    for question in test_questions:
        answer = chatbot.answer_question(question)
        print(f"\n🤖 Assistant: {answer}")
        print("\n" + "="*60 + "\n")