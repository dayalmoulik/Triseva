import warnings
warnings.filterwarnings("ignore")

import os
from langchain_core.tools import tool
from tavily import TavilyClient

_tavily_client = None

def _get_tavily():
    global _tavily_client
    if _tavily_client is None:
        _tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY") or os.getenv("Tavily_API_Key"))
    return _tavily_client


@tool
def web_search_tool(query: str) -> str:
    """
    Search the web for current information about government schemes,
    medical guidelines, or agriculture advisories not found in the knowledge base.
    Use this when RAG retrieval returns insufficient or outdated information.
    """
    try:
        client = _get_tavily()
        results = client.search(
            query=query,
            max_results=3,
            search_depth="basic",
        )

        if not results.get("results"):
            return "No web results found."

        output = ""
        for i, r in enumerate(results["results"], 1):
            output += f"[Web Result {i}: {r['title']}]\n"
            output += f"URL: {r['url']}\n"
            output += f"{r.get('content', 'No content available')[:500]}\n\n"

        return output.strip()

    except Exception as e:
        return f"Web search failed: {str(e)}"