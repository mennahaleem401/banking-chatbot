"""
Streamlit Banking Chatbot Application
Main application file with intelligent chat interface
"""

import streamlit as st
import os
from typing import Dict, Any
import pandas as pd
from data_processor import BankingDataProcessor, create_sample_data
from vector_store import initialize_vector_store
from llama_model import get_llama_model
from rag_pipeline import BankingRAGPipeline, IntelligentBankingAnalyzer


# Page configuration
st.set_page_config(
    page_title="GlobalTrust Bank Assistant",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional banking interface
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .user-info {
        background-color: black;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #1f77b4;
    }
    .security-notice {
        background-color: black;
        border-left: 4px solid #ffc107;
        padding: 0.75rem;
        margin-top: 1rem;
        border-radius: 0.25rem;
        color: #856404;
    }
    .financial-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .quick-action {
        background-color: #e8f4fd;
        border: 1px solid #b6e0fe;
        border-radius: 0.5rem;
        padding: 0.75rem;
        margin: 0.25rem 0;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    .quick-action:hover {
        background-color: #b6e0fe;
        transform: translateY(-2px);
    }
    .chat-message-user {
        background-color: black;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #1976d2;
    }
    .chat-message-assistant {
        background-color: black;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #7b1fa2;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def initialize_system():
    """Initialize the banking chatbot system"""
    
    with st.spinner("🔄 Initializing Banking Chatbot System..."):
        try:
            # Create sample data if needed
            data_dir = os.path.join(os.path.dirname(__file__), "uploads")
            if not os.path.exists(data_dir) or not os.listdir(data_dir):
                st.info("📁 Creating sample banking data...")
                create_sample_data()
            
            # Define file paths
            accounts_path = os.path.join(data_dir, "accounts.csv")
            transactions_path = os.path.join(data_dir, "transactions.csv")
            users_path = os.path.join(data_dir, "users.csv")
            user_financials_path = os.path.join(data_dir, "user_financials.csv")
            
            # Load and process data
           
            processor = BankingDataProcessor(
                accounts_path=accounts_path,
                transactions_path=transactions_path,
                users_path=users_path,
                user_financials_path=user_financials_path
            )
            
            documents = processor.process_all()
            
            # Initialize vector store
            
            vector_store = initialize_vector_store(
                documents,
                persist_directory="./chroma_db",
                force_recreate=False
            )
            
            # Initialize Llama model
            
            
            # Get API key from environment or Streamlit secrets
            api_key = None
            try:
                if hasattr(st, 'secrets') and 'GROQ_API_KEY' in st.secrets:
                    api_key = st.secrets['GROQ_API_KEY']
                    
            except:
                pass
                
            if not api_key:
                api_key = os.getenv('GROQ_API_KEY')
                if api_key:
                    st.success("✅ Loaded API key from environment variable")
            
            use_mock = not api_key
            if use_mock:
                st.warning("⚠️ GROQ_API_KEY not found. Using intelligent mock model.")
            
            # Use a supported Groq model name (old 'llama3-8b-8192' was decommissioned)
            llama_model = get_llama_model(
                use_mock=use_mock,
                model_type="groq",
                model_name="llama-3.3-70b-versatile",
                api_key=api_key
            )
            
            # Create RAG pipeline
            rag_pipeline = BankingRAGPipeline(vector_store, llama_model, retrieval_k=5)
            
            # Create intelligent analyzer
            analyzer = IntelligentBankingAnalyzer(rag_pipeline)
            
            # Load user data for display
            users_df = pd.read_csv(users_path)
            
            
            
            return rag_pipeline, analyzer, processor, users_df
            
        except Exception as e:
            st.error(f"❌ Error initializing system: {str(e)}")
            st.stop()


def get_user_info(users_df: pd.DataFrame, user_id: int) -> Dict[str, Any]:
    """Get user information"""
    user_row = users_df[users_df['id'] == user_id].iloc[0]
    return {
        'username': user_row['username'],
        'email': user_row['email'],
        'role': user_row['role'],
        'is_active': user_row['isActive'] == 1
    }


def display_user_info(user_info: Dict[str, Any]):
    """Display user information in sidebar"""
    st.sidebar.markdown("### 👤 Current User")
    st.sidebar.markdown(f"""
    <div class="user-info">
        <strong>Username:</strong> {user_info['username']}<br>
        <strong>Email:</strong> {user_info['email']}<br>
        <strong>Role:</strong> {user_info['role']}<br>
        <strong>Status:</strong> {'🟢 Active' if user_info['is_active'] else '🔴 Inactive'}
    </div>
    """, unsafe_allow_html=True)


def display_financial_overview(analyzer, user_id: int):
    """Display financial overview in sidebar"""
    try:
        summary = analyzer.rag_pipeline.get_user_financial_summary(user_id)
        
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 💰 Financial Overview")
        
        st.sidebar.markdown(f"""
        <div class="financial-card">
            <strong>Total Balance:</strong><br>
            <h3>${summary.get('total_balance', 0):,.2f}</h3>
            <small>Across {summary.get('account_count', 0)} accounts</small>
        </div>
        """, unsafe_allow_html=True)
        
        # Recent activity
        st.sidebar.markdown(f"**Recent Transactions:** {summary.get('recent_transactions', 0)}")
        
    except Exception as e:
        st.sidebar.warning("Financial overview temporarily unavailable")


def main():
    """Main application function"""
    
    # Header
    st.markdown('<div class="main-header">🏦 GlobalTrust Bank AI Assistant</div>', 
                unsafe_allow_html=True)
    
    # Initialize system
    try:
        rag_pipeline, analyzer, processor, users_df = initialize_system()
    except Exception as e:
        st.error(f"❌ Error initializing system: {str(e)}")
        st.stop()
    
    # Sidebar - User Selection
    st.sidebar.title("🔐 User Authentication")
    
    # Get list of users
    user_options = {
        f"{row['id']}: {row['username']} ({row['role']})": row['id'] 
        for _, row in users_df.iterrows()
    }
    
    selected_user = st.sidebar.selectbox(
        "Select User",
        options=list(user_options.keys()),
        index=0
    )
    
    user_id = user_options[selected_user]
    
    # Display user info and financial overview
    user_info = get_user_info(users_df, user_id)
    display_user_info(user_info)
    display_financial_overview(analyzer, user_id)
    
    # Sidebar - AI Settings
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ AI Settings")
    
    temperature = st.sidebar.slider(
        "Response Creativity",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.1,
        help="Lower values for factual accuracy, higher for creative responses"
    )
    
    # Sidebar - Intelligent Features
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🚀 Smart Actions")
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.button("💰 Balance", use_container_width=True):
            st.session_state.quick_query = "What is my current account balance?"
        
        if st.button("📊 Transactions", use_container_width=True):
            st.session_state.quick_query = "Show me my recent transactions and spending patterns"
    
    with col2:
        if st.button("💼 Profile", use_container_width=True):
            st.session_state.quick_query = "What is my financial profile and credit information?"
        
        if st.button("🎯 Advice", use_container_width=True):
            st.session_state.quick_query = "Give me personalized financial advice based on my situation"
    
    if st.sidebar.button("🔄 Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    # Advanced analytics section
    with st.sidebar.expander("📈 Advanced Analytics"):
        if st.button("Analyze Spending Patterns"):
            with st.spinner("Analyzing your spending..."):
                analysis = analyzer.analyze_spending_patterns(user_id)
                st.write("**Spending Analysis:**")
                st.write(f"Total Spent: ${analysis.get('total_spent', 0):,.2f}")
                st.write(f"Total Earned: ${analysis.get('total_earned', 0):,.2f}")
                st.write(f"Net Cash Flow: ${analysis.get('net_flow', 0):,.2f}")
        
        if st.button("Generate Financial Advice"):
            with st.spinner("Generating personalized advice..."):
                advice = analyzer.generate_financial_advice(user_id)
                st.write("**Personalized Advice:**")
                st.write(advice)
    
    # Main chat interface
    st.markdown("### 💬 Intelligent Banking Assistant")
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "user_id" not in st.session_state or st.session_state.user_id != user_id:
        st.session_state.user_id = user_id
        st.session_state.messages = []
        # Add intelligent welcome message
        welcome_msg = f"""Hello {user_info['username']}! I'm your AI banking assistant at GlobalTrust Bank. 

I can help you with:
• Account balances and transactions
• Financial profile information  
• Spending pattern analysis
• Personalized financial advice
• Banking inquiries and support

How can I assist you with your banking needs today?"""
        st.session_state.messages.append({
            "role": "assistant",
            "content": welcome_msg
        })
    
    # Display chat messages with enhanced styling
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f"""
            <div class="chat-message-user">
                <strong>You:</strong><br>
                {message["content"]}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-message-assistant">
                <strong>Assistant:</strong><br>
                {message["content"]}
            </div>
            """, unsafe_allow_html=True)
    
    # Handle quick query
    if "quick_query" in st.session_state:
        prompt = st.session_state.quick_query
        del st.session_state.quick_query
    else:
        prompt = st.chat_input("Ask about your accounts, transactions, or financial advice...")
    
    # Process user input
    if prompt:
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Generate response
        with st.spinner("🔍 Analyzing your request with AI..."):
            try:
                # Get intelligent response from RAG pipeline
                result = rag_pipeline.classify_and_handle_query(
                    user_id=user_id,
                    question=prompt
                )
                
                response = result['response']
                
                # Add to message history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response
                })
                
                # Show intelligent features in expander
                with st.expander("🧠 AI Analysis Details"):
                    st.write(f"**Detected Intent:** {result['intent']['type']}")
                    st.write(f"**Urgency Level:** {result['intent']['urgency']}")
                    
                    if st.checkbox("Show Retrieved Context"):
                        st.text_area("Context", result['context'], height=200)
                
                st.rerun()
                
            except Exception as e:
                error_msg = f"❌ Error generating response: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })
                st.rerun()
    
    # Security notice
    st.markdown("""
    <div class="security-notice">
        🔒 <strong>Security Notice:</strong> This is a secure banking interface. 
        Your data is protected and never shared. All AI responses are based solely on your banking information.
    </div>
    """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
        GlobalTrust Bank AI Assistant | Powered by Advanced RAG & Llama Technology<br>
        For support, contact: support@globaltrustbank.com | Secure • Intelligent • Personalized
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()