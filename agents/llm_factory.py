"""
TriSeva LLM Factory & Multi-Provider Resilient Failover Module.

Instantiates primary LLM model backends (Sarvam-105B Indic LLM) and attaches automatic
resilient fallbacks (Groq Llama-3.3-70B, OpenAI GPT-4o-mini, Anthropic Claude Haiku)
to guarantee high availability and sub-second failover.
"""

import os
from langchain_openai import ChatOpenAI

def get_llm(temperature: float = 0.2, max_tokens: int = 1024, timeout: int = 20):
    """Factory function to retrieve LLM instance with automated multi-provider fallbacks.

    Configures primary provider based on `LLM_PROVIDER` environment variable ('sarvam', 'groq', 'openai')
    and chains fallback models via LangChain's `.with_fallbacks()`.

    Args:
        temperature (float, optional): Sampling temperature (0.0 for deterministic output). Defaults to 0.2.
        max_tokens (int, optional): Maximum token generation limit. Defaults to 1024.
        timeout (int, optional): Per-request HTTP timeout in seconds. Defaults to 20.

    Returns:
        BaseChatModel: Configured LLM model instance with chained fallbacks.

    Raises:
        ValueError: If no valid LLM provider API key is present in environment variables.
    """
    primary_provider = (os.getenv("LLM_PROVIDER") or os.getenv("LLM_Provider") or "sarvam").lower()
    
    fallbacks = []

    # 1. Prepare Groq Fallback if key available
    groq_key = os.getenv("GROQ_API_KEY") or os.getenv("Groq_API_Key")
    if groq_key and primary_provider != "groq":
        try:
            from langchain_groq import ChatGroq
            fallbacks.append(ChatGroq(
                model="llama-3.3-70b-versatile",
                api_key=groq_key,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=12,
            ))
        except Exception:
            pass

    # 2. Prepare OpenAI Fallback if key available
    openai_key = os.getenv("OPENAI_API_KEY") or os.getenv("OpenAI_API_Key")
    if openai_key and primary_provider != "openai":
        try:
            fallbacks.append(ChatOpenAI(
                model="gpt-4o-mini",
                api_key=openai_key,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=12,
            ))
        except Exception:
            pass

    # 3. Build Primary LLM instance
    primary_llm = None
    if primary_provider == "sarvam":
        api_key = os.getenv("SARVAM_API_KEY") or os.getenv("Sarvam_API_Key")
        if api_key:
            primary_llm = ChatOpenAI(
                model="sarvam-105b",
                openai_api_key=api_key,
                openai_api_base="https://api.sarvam.ai/v1",
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )
    elif primary_provider == "groq" and groq_key:
        from langchain_groq import ChatGroq
        primary_llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=groq_key,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
    elif primary_provider == "openai" and openai_key:
        primary_llm = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=openai_key,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )

    if not primary_llm:
        if fallbacks:
            return fallbacks[0]
        # Anthropic Claude fallback
        api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("Anthropic_API_Key")
        if api_key:
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(
                model="claude-3-5-haiku-latest",
                api_key=api_key,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )
        raise ValueError("No valid LLM API key configured.")

    if fallbacks:
        return primary_llm.with_fallbacks(fallbacks)
    return primary_llm
