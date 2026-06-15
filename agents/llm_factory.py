import os
from langchain_openai import ChatOpenAI

def get_llm(temperature=0.2, max_tokens=1024, timeout=30):
    """Factory function to retrieve LLM backend, supporting Sarvam, OpenAI, Groq, and Anthropic."""
    provider = (os.getenv("LLM_PROVIDER") or os.getenv("LLM_Provider") or "sarvam").lower()
    
    if provider == "sarvam":
        api_key = os.getenv("SARVAM_API_KEY") or os.getenv("Sarvam_API_Key")
        if not api_key:
            raise ValueError(
                "SARVAM_API_KEY is not configured in the Space Secrets. "
                "Please go to your Hugging Face Space Settings, add your SARVAM_API_KEY as a Secret, "
                "and restart/rebuild the Space."
            )
            
        return ChatOpenAI(
            model="sarvam-105b",
            openai_api_key=api_key,
            openai_api_base="https://api.sarvam.ai/v1",
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
    elif provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OpenAI_API_Key")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY is not configured in the Space Secrets. "
                "Please go to your Hugging Face Space Settings and add your OPENAI_API_KEY as a Secret."
            )
        return ChatOpenAI(
            model="gpt-4o-mini",
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
    elif provider == "groq":
        api_key = os.getenv("GROQ_API_KEY") or os.getenv("Groq_API_Key")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY is not configured in the Space Secrets. "
                "Please go to your Hugging Face Space Settings and add your GROQ_API_KEY as a Secret."
            )
        from langchain_groq import ChatGroq
        return ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
    else:
        # Anthropic Claude fallback
        from langchain_anthropic import ChatAnthropic
        api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("Anthropic_API_Key")
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY is not configured in the Space Secrets. "
                "Please go to your Hugging Face Space Settings and add your ANTHROPIC_API_KEY as a Secret."
            )
        return ChatAnthropic(
            model="claude-3-5-haiku-latest",
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
