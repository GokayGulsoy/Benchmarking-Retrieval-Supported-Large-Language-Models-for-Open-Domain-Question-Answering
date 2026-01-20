import os
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

# function to return an instance of LLM model based on given model name
def get_llm_model(model_name, temperature=0.3):
    """
    Factory function to return the correct LangChain chat model instance
    based on the model name string.
    
    Supported prefixes: "gpt", "claude", "gemini"
    """
    model_name_lower = model_name.lower()
    
    if "gpt" in model_name_lower:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set.")

        return ChatOpenAI(model=model_name, api_key=api_key, temperature=temperature)

    elif "claude" in model_name_lower:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set.")

        return ChatAnthropic(model=model_name, api_key=api_key, temperature=temperature)

    elif "gemini" in model_name_lower:
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set.")
        
        return ChatGoogleGenerativeAI(model=model_name, api_key=api_key, temperature=temperature)

    else:
        raise ValueError(f"Unknown model provider for model name: {model_name}. Please use a name containing 'gpt', 'claude', or 'gemini'")
