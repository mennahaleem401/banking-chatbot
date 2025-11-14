"""
RAG Pipeline for Banking Chatbot
Retrieval-Augmented Generation pipeline for intelligent banking responses
"""

from typing import Dict, Any, List, Optional
from vector_store import BankingVectorStore
from llama_model import GroqLlamaModel, MockLlamaModel
import re


class BankingRAGPipeline:
    """RAG pipeline for banking-specific queries"""
    
    def __init__(self, 
                 vector_store: BankingVectorStore,
                 llama_model: Any,
                 retrieval_k: int = 5):
        """
        Initialize RAG pipeline
        
        Args:
            vector_store: Initialized vector store instance
            llama_model: Initialized LLM model
            retrieval_k: Number of documents to retrieve
        """
        self.vector_store = vector_store
        self.llama_model = llama_model
        self.retrieval_k = retrieval_k
        
    def _retrieve_context(self, question: str, user_id: int) -> str:
        """
        Retrieve relevant context for the question
        
        Args:
            question: User question
            user_id: User ID for filtering
            
        Returns:
            Retrieved context as string
        """
        # Search for similar documents
        results = self.vector_store.search_similar(
            query=question,
            user_id=user_id,
            n_results=self.retrieval_k
        )
        
        if not results:
            return "No relevant banking information found."
        
        # Format context
        context_parts = []
        for i, result in enumerate(results):
            context_parts.append(f"[Document {i+1}]: {result['content']}")
        
        return "\n\n".join(context_parts)
    
    def _create_enhanced_prompt(self, 
                               question: str, 
                               context: str, 
                               user_id: int) -> str:
        """
        Create enhanced prompt with banking context
        
        Args:
            question: User question
            context: Retrieved context
            user_id: User ID
            
        Returns:
            Enhanced prompt for LLM
        """
        prompt = f"""You are GlobalTrust Bank's intelligent banking assistant. Your role is to provide accurate, helpful, and secure banking information to users.

USER ID: {user_id}
CONTEXT FROM BANKING RECORDS:
{context}

USER QUESTION: {question}

INSTRUCTIONS:
1. Use ONLY the information from the banking context above to answer the question
2. If specific numbers are in the context, use them precisely
3. Be professional, clear, and helpful
4. If information is missing from the context, acknowledge this and suggest what the user can do
5. Never invent or hallucinate financial information
6. Focus on being accurate rather than creative
7. For balance inquiries, reference the exact amounts from accounts
8. For transaction questions, reference the specific transactions shown
9. For financial profile questions, use the income and credit information provided

Please provide a helpful response based on the available banking information:"""

        return prompt
    
    def _detect_query_intent(self, question: str) -> Dict[str, Any]:
        """
        Detect the intent of the user query
        
        Args:
            question: User question
            
        Returns:
            Intent analysis
        """
        question_lower = question.lower()
        
        intent = {
            'type': 'general',
            'urgency': 'normal',
            'needs_calculation': False,
            'sensitive': False
        }
        
        # Detect query type
        if any(word in question_lower for word in ['balance', 'how much', 'amount']):
            intent['type'] = 'balance_inquiry'
        elif any(word in question_lower for word in ['transaction', 'payment', 'transfer']):
            intent['type'] = 'transaction_query'
        elif any(word in question_lower for word in ['account', 'savings', 'checking']):
            intent['type'] = 'account_info'
        elif any(word in question_lower for word in ['income', 'salary', 'financial', 'credit']):
            intent['type'] = 'financial_profile'
        elif any(word in question_lower for word in ['help', 'support', 'assist']):
            intent['type'] = 'customer_support'
        
        # Detect urgency
        if any(word in question_lower for word in ['urgent', 'emergency', 'immediately', 'asap']):
            intent['urgency'] = 'high'
        
        # Detect if calculation might be needed
        if any(word in question_lower for word in ['total', 'sum', 'calculate', 'how many']):
            intent['needs_calculation'] = True
        
        # Detect sensitive information
        if any(word in question_lower for word in ['password', 'pin', 'social security', 'ssn']):
            intent['sensitive'] = True
        
        return intent
    
    def _generate_intelligent_response(self, 
                                     prompt: str, 
                                     intent: Dict[str, Any]) -> str:
        """
        Generate response using LLM with intent awareness
        
        Args:
            prompt: Enhanced prompt
            intent: Query intent analysis
            
        Returns:
            Generated response
        """
        # Adjust generation parameters based on intent
        temperature = 0.3  # Lower temperature for factual accuracy
        max_length = 512
        
        if intent['type'] == 'customer_support':
            temperature = 0.7  # Slightly more creative for support questions
        if intent['urgency'] == 'high':
            max_length = 256  # Shorter responses for urgent queries
        
        if intent['sensitive']:
            return "I cannot provide assistance with sensitive security information like passwords or PINs. Please contact our security team directly for such inquiries."
        
        # Generate response
        response = self.llama_model.generate_response(
            prompt=prompt,
            temperature=temperature,
            max_length=max_length,
            top_p=0.9
        )
        
        return response
    
    def classify_and_handle_query(self, user_id: int, question: str) -> Dict[str, Any]:
        """
        Main method to handle user queries with RAG
        
        Args:
            user_id: User ID
            question: User question
            
        Returns:
            Response dictionary with answer and metadata
        """
        print(f"🔄 Processing query for user {user_id}: {question}")
        
        # Step 1: Detect query intent
        intent = self._detect_query_intent(question)
        print(f"📊 Detected intent: {intent}")
        
        # Step 2: Retrieve relevant context
        context = self._retrieve_context(question, user_id)
        print(f"📚 Retrieved context with {len(context.splitlines())} lines")
        
        # Step 3: Create enhanced prompt
        prompt = self._create_enhanced_prompt(question, context, user_id)
        
        # Step 4: Generate intelligent response
        response = self._generate_intelligent_response(prompt, intent)
        
        # Step 5: Prepare result
        result = {
            'response': response,
            'context': context,
            'intent': intent,
            'user_id': user_id,
            'question': question
        }
        
        print("✅ Query processed successfully")
        return result
    
    def get_user_financial_summary(self, user_id: int) -> Dict[str, Any]:
        """
        Generate a financial summary for the user
        
        Args:
            user_id: User ID
            
        Returns:
            Financial summary
        """
        # Get all user documents
        user_docs = self.vector_store.get_user_documents(user_id, limit=20)
        
        accounts = []
        transactions = []
        financial_profile = None
        
        for doc in user_docs:
            metadata = doc['metadata']
            if metadata['type'] == 'account':
                accounts.append(metadata)
            elif metadata['type'] == 'transaction':
                transactions.append(metadata)
            elif metadata['type'] == 'financial_profile':
                financial_profile = metadata
        
        # Calculate totals
        total_balance = sum(acc.get('balance', 0) for acc in accounts)
        recent_transactions_count = len(transactions)
        
        summary = {
            'total_balance': total_balance,
            'account_count': len(accounts),
            'recent_transactions': recent_transactions_count,
            'accounts': accounts,
            'financial_profile': financial_profile
        }
        
        return summary


class IntelligentBankingAnalyzer:
    """Advanced analytics for banking intelligence"""
    
    def __init__(self, rag_pipeline: BankingRAGPipeline):
        self.rag_pipeline = rag_pipeline
    
    def analyze_spending_patterns(self, user_id: int) -> Dict[str, Any]:
        """Analyze user spending patterns"""
        user_docs = self.rag_pipeline.vector_store.get_user_documents(user_id, limit=50)
        
        transactions = [doc for doc in user_docs if doc['metadata']['type'] == 'transaction']
        
        if not transactions:
            return {"error": "No transaction data available"}
        
        # Simple analysis
        total_spent = sum(t['metadata'].get('amount', 0) for t in transactions if t['metadata'].get('amount', 0) < 0)
        total_earned = sum(t['metadata'].get('amount', 0) for t in transactions if t['metadata'].get('amount', 0) > 0)
        
        return {
            'total_spent': abs(total_spent),
            'total_earned': total_earned,
            'transaction_count': len(transactions),
            'net_flow': total_earned + total_spent,  # total_earned is positive, total_spent is negative
            'analysis': "Basic spending pattern analysis completed"
        }
    
    def generate_financial_advice(self, user_id: int) -> str:
        """Generate personalized financial advice"""
        summary = self.rag_pipeline.get_user_financial_summary(user_id)
        spending_analysis = self.analyze_spending_patterns(user_id)
        
        prompt = f"""
        Based on the following financial summary and spending analysis, provide 2-3 practical financial advice suggestions:
        
        FINANCIAL SUMMARY:
        - Total Balance: ${summary.get('total_balance', 0):,.2f}
        - Number of Accounts: {summary.get('account_count', 0)}
        - Recent Transactions: {summary.get('recent_transactions', 0)}
        
        SPENDING ANALYSIS:
        - Total Spent: ${spending_analysis.get('total_spent', 0):,.2f}
        - Total Earned: ${spending_analysis.get('total_earned', 0):,.2f}
        - Net Cash Flow: ${spending_analysis.get('net_flow', 0):,.2f}
        
        Please provide actionable, personalized financial advice:"""
        
        advice = self.rag_pipeline.llama_model.generate_response(
            prompt=prompt,
            temperature=0.5,
            max_length=300
        )
        
        return advice


if __name__ == "__main__":
    # Test the RAG pipeline
    from data_processor import BankingDataProcessor, create_sample_data
    from vector_store import initialize_vector_store
    from llama_model import get_llama_model
    
    # Create and process sample data
    create_sample_data()
    processor = BankingDataProcessor(
        accounts_path="uploads/accounts.csv",
        transactions_path="uploads/transactions.csv",
        users_path="uploads/users.csv", 
        user_financials_path="uploads/user_financials.csv"
    )
    documents = processor.process_all()
    
    # Initialize components
    vector_store = initialize_vector_store(documents)
    llama_model = get_llama_model(use_mock=True)
    
    # Create RAG pipeline
    rag_pipeline = BankingRAGPipeline(vector_store, llama_model)
    
    # Test query
    result = rag_pipeline.classify_and_handle_query(
        user_id=1, 
        question="What is my current account balance?"
    )
    
    print("🧪 Test Results:")
    print(f"Question: {result['question']}")
    print(f"Intent: {result['intent']}")
    print(f"Response: {result['response']}")
    
    # Test analytics
    analyzer = IntelligentBankingAnalyzer(rag_pipeline)
    spending_analysis = analyzer.analyze_spending_patterns(1)
    print(f"Spending Analysis: {spending_analysis}")