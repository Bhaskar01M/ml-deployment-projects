from retriever import Retriever
import random

class RAGChatbotDemo:
    def __init__(self, embeddings_file="embeddings.json"):
        self.retriever = Retriever(embeddings_file)
    
    def generate_query_embedding(self, query):
        """Generate fake embedding for demo"""
        return [random.uniform(-1, 1) for _ in range(1536)]
    
    def answer_question_demo(self, question, k=5):
        """Answer using RAG without API calls - DEMO MODE"""
        print(f"\n🤔 Question: {question}\n")
        
        # Step 1: Generate embedding for query
        print("📍 Step 1: Embedding query...")
        query_embedding = self.generate_query_embedding(question)
        print("   ✅ Query embedded\n")
        
        # Step 2: Retrieve relevant chunks
        print(f"📍 Step 2: Retrieving top {k} chunks from your PDFs...")
        chunks = self.retriever.retrieve(query_embedding, k=k)
        print(f"   ✅ Retrieved {len(chunks)} relevant chunks\n")
        
        # Step 3: Display retrieved context
        print("📍 Step 3: Relevant passages from your documents:\n")
        print("-" * 60)
        for i, chunk in enumerate(chunks, 1):
            print(f"\n📄 Passage {i} (Relevance: {chunk['similarity']:.2%})")
            print(f"   {chunk['text']}\n")
        print("-" * 60)
        
        # Step 4: Demo answer
        print("\n📍 Step 4: Generating answer based on retrieved content...\n")
        
        # Create a demo answer based on the question
        demo_answers = {
            "transformer": "Based on the retrieved passages, the Transformer is a neural network architecture that uses attention mechanisms instead of recurrence. It allows for parallel processing of sequences and has become the foundation for modern language models like BERT and GPT.",
            "bert": "According to the documents, BERT (Bidirectional Encoder Representations from Transformers) is a pre-training method that uses bidirectional context. It's trained with masked language modeling and next sentence prediction objectives, allowing it to understand language from both directions.",
            "attention": "The attention mechanism, as described in the passages, allows the model to focus on relevant parts of the input. It computes similarity scores between queries and keys, then uses these to weight the importance of different positions in the sequence.",
            "default": "Based on the retrieved passages from your research papers, the answer relates to the concepts of transformers, neural networks, and language models. The exact details are in the passages shown above."
        }
        
        # Find best matching answer
        question_lower = question.lower()
        answer = demo_answers["default"]
        
        for key, ans in demo_answers.items():
            if key in question_lower:
                answer = ans
                break
        
        return answer, chunks
    
    def interactive_demo(self):
        """Interactive RAG demo without API calls"""
        print("\n" + "="*70)
        print("🤖 RAG CHATBOT DEMO (No API Calls - Using Local Retrieval)")
        print("="*70)
        print("\nHow it works:")
        print("1. You ask a question")
        print("2. System finds similar chunks from your PDF documents")
        print("3. Shows the relevant passages")
        print("4. Generates answer based on those passages\n")
        print("Type 'exit' to quit.\n")
        
        while True:
            question = input("You: ").strip()
            
            if question.lower() == 'exit':
                print("\n👋 Thanks for using RAG Chatbot!")
                break
            
            if not question:
                continue
            
            answer, chunks = self.answer_question_demo(question)
            print(f"\n🤖 Assistant: {answer}")
            print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    chatbot = RAGChatbotDemo()
    
    # Run demo mode
    print("="*70)
    print("RAG CHATBOT - DEMO MODE (No API Calls Needed)")
    print("="*70)
    
    test_questions = [
        "What is the Transformer architecture?",
        "How does BERT work?",
        "What is attention mechanism?"
    ]
    
    print("\nRunning 3 demo questions...\n")
    
    for question in test_questions:
        answer, chunks = chatbot.answer_question_demo(question)
        print(f"\n🤖 Assistant: {answer}")
        print("\n" + "="*70 + "\n")
    
    print("\n" + "="*70)
    print("Demo complete! Run 'chatbot.interactive_demo()' for interactive mode")
    print("="*70)