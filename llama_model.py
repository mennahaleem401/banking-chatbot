"""
Llama Model Integration Module for Banking Chatbot
Handles Llama model loading and response generation with Groq Cloud
"""

import os
from typing import Optional, Dict, Any
from groq import Groq


class GroqLlamaModel:
    """Wrapper for Groq Cloud Llama model integration"""
    
    def __init__(self, 
                 api_key: str,
                 model_name: str = "llama-3.3-70b-versatile",
                 base_url: Optional[str] = None):
        """
        Initialize Groq Llama model
        
        Args:
            api_key: Groq API key
            model_name: Groq model name
            base_url: Optional base URL for API (defaults to Groq cloud)
        """
        self.model_name = model_name
        self.client = Groq(api_key=api_key, base_url=base_url)
        self.loaded = True
        
    def generate_response(self, 
                         prompt: str, 
                         max_length: int = 512,
                         temperature: float = 0.7,
                         top_p: float = 0.9) -> str:
        """
        Generate response using Groq API
        
        Args:
            prompt: Input prompt
            max_length: Maximum length of generated response
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            
        Returns:
            Generated response text
        """
        try:
            response = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model=self.model_name,
                temperature=temperature,
                max_tokens=max_length,
                top_p=top_p,
                stream=False
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            # If the model was decommissioned, try to map to a supported replacement and retry once
            err_str = str(e)
            print(f"Error calling Groq API: {err_str}")

            # detect decommission message/code
            if 'decommission' in err_str.lower() or 'model_decommissioned' in err_str.lower():
                # try to resolve alias -> key -> actual model name
                alias_key = GROQ_ALIASES.get(self.model_name, None)
                fallback_key = alias_key or GROQ_ALIASES.get(self.model_name.lower(), None)
                if fallback_key:
                    replacement = GROQ_MODELS.get(fallback_key)
                    if replacement:
                        print(f"Attempting fallback to replacement model '{replacement}'")
                        self.model_name = replacement
                        try:
                            response = self.client.chat.completions.create(
                                messages=[{"role": "user", "content": prompt}],
                                model=self.model_name,
                                temperature=temperature,
                                max_tokens=max_length,
                                top_p=top_p,
                                stream=False
                            )
                            return response.choices[0].message.content.strip()
                        except Exception as e2:
                            print(f"Fallback attempt also failed: {e2}")
                            return f"Error generating response after fallback: {str(e2)}"

            return f"Error generating response: {err_str}"


class MockLlamaModel:
    """Mock Llama model for testing without actual model"""
    
    def __init__(self):
        """Initialize mock model"""
        print("Initializing Mock Llama Model (for testing)")
        self.loaded = True
    
    def generate_response(self, 
                         prompt: str, 
                         max_length: int = 512,
                         temperature: float = 0.7,
                         top_p: float = 0.9) -> str:
        """
        Generate mock response based on prompt
        
        Args:
            prompt: Input prompt
            max_length: Maximum length (ignored in mock)
            temperature: Temperature (ignored in mock)
            top_p: Top-p (ignored in mock)
            
        Returns:
            Mock response
        """
        # Simple rule-based responses for testing
        prompt_lower = prompt.lower()
        
        if "balance" in prompt_lower:
            return "Based on the account information provided, your current account balance is shown in the context above. Please refer to the specific account details for the exact amount."
        
        elif "transaction" in prompt_lower:
            return "I can see your recent transactions in the context. The transactions show various credits and debits to your accounts. Would you like me to provide more details about any specific transaction?"
        
        elif "income" in prompt_lower or "financial" in prompt_lower:
            return "According to your financial profile, your monthly income and credit score information is available in the context above. This information helps us provide you with better banking services."
        
        elif "account" in prompt_lower:
            return "Your account information is displayed in the context. You have access to various account types with different balances. Is there a specific account you'd like to know more about?"
        
        else:
            return "I'm here to help you with your banking needs. Based on the information provided in the context, I can assist you with questions about your accounts, transactions, balances, and financial profile. How can I help you today?"


class OllamaModel:
    """Ollama model for local deployment (backup option)"""
    
    def __init__(self, model_name: str = "llama2"):
        """
        Initialize Ollama model
        
        Args:
            model_name: Ollama model name
        """
        try:
            from langchain_community.llms import Ollama
            self.model = Ollama(model=model_name)
            self.loaded = True
            print(f"Ollama model '{model_name}' loaded successfully!")
        except Exception as e:
            print(f"Error loading Ollama model: {e}")
            self.loaded = False
    
    def generate_response(self, 
                         prompt: str, 
                         max_length: int = 512,
                         temperature: float = 0.7,
                         top_p: float = 0.9) -> str:
        """Generate response using Ollama"""
        if not self.loaded:
            return "Error: Ollama model not loaded"
        
        try:
            response = self.model.invoke(prompt)
            return response.strip()
        except Exception as e:
            return f"Error generating response with Ollama: {str(e)}"


def get_llama_model(use_mock: bool = False, 
                   model_type: str = "groq",
                   model_name: str = "llama-3.3-70b-versatile",
                   api_key: Optional[str] = None) -> Any:
    """
    Factory function to get appropriate Llama model
    
    Args:
        use_mock: If True, return mock model for testing
        model_type: Type of model loading ('groq', 'ollama', 'mock')
        model_name: Model name or path
        api_key: Groq API key (required for groq type)
        
    Returns:
        Initialized model object
    """
    if use_mock:
        print("Using Mock Llama Model for testing")
        return MockLlamaModel()
    
    if model_type == "groq":
        # Get API key from environment if not provided
        if api_key is None:
            api_key = os.getenv("GROQ_API_KEY")
            
        if not api_key:
            print("Warning: GROQ_API_KEY not found. Falling back to mock model.")
            return MockLlamaModel()
        
        try:
            print(f"Initializing Groq model: {model_name}")
            model = GroqLlamaModel(api_key=api_key, model_name=model_name)
            print("Groq model initialized successfully!")
            return model
        except Exception as e:
            print(f"Error initializing Groq model: {e}. Falling back to mock model.")
            return MockLlamaModel()
    
    elif model_type == "ollama":
        try:
            print(f"Initializing Ollama model: {model_name}")
            model = OllamaModel(model_name=model_name)
            if model.loaded:
                return model
            else:
                print("Falling back to mock model")
                return MockLlamaModel()
        except Exception as e:
            print(f"Error initializing Ollama model: {e}. Falling back to mock model.")
            return MockLlamaModel()
    
    elif model_type == "mock":
        return MockLlamaModel()
    
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Supported types: 'groq', 'ollama', 'mock'")


# Available Groq models
GROQ_MODELS = {
    "llama3-8b": "llama-3.3-70b-versatile",
    "llama3-70b": "llama-3.3-70b-versatile", 
    "mixtral": "mixtral-8x7b-32768",
    "gemma": "gemma-7b-it"
}

# Backwards-compatible aliases for previously used model names
GROQ_ALIASES = {
    "llama3-8b-8192": "llama3-8b",
    "llama3-8b-32768": "llama3-8b",
}


def get_available_models() -> Dict[str, str]:
    """Get available Groq models"""
    return GROQ_MODELS.copy()


if __name__ == "__main__":
    # Test the models
    print("Testing Llama Model Integration...")
    
    # Test mock model
    print("\n1. Testing Mock Model:")
    mock_model = MockLlamaModel()
    test_prompt = "What is my account balance?"
    response = mock_model.generate_response(test_prompt)
    print(f"Prompt: {test_prompt}")
    print(f"Response: {response}")
    
    # Test Groq model if API key available
    groq_api_key = os.getenv("GROQ_API_KEY")
    if groq_api_key:
        print("\n2. Testing Groq Model:")
        try:
            groq_model = GroqLlamaModel(api_key=groq_api_key)
            response = groq_model.generate_response(test_prompt, max_length=100)
            print(f"Prompt: {test_prompt}")
            print(f"Response: {response}")
        except Exception as e:
            print(f"Groq test failed: {e}")
    else:
        print("\n2. Groq API key not found, skipping Groq test")
    
    print("\n3. Testing model factory:")
    model = get_llama_model(use_mock=True)
    response = model.generate_response("Test prompt")
    print(f"Factory test response: {response}")