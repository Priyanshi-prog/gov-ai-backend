import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.schema import SystemMessage, HumanMessage

def get_ai_recommendation(user_job, state, user_query):
    # 1. THE LIBRARIAN: Searches live official gov sites
    search = TavilySearchResults(
        max_results=3, 
        include_domains=["gov.in", "nic.in"] # Gov only
    )
    
    search_query = f"official government scheme policy for {user_job} in {state} {user_query} 2026"
    live_gov_context = search.run(search_query)

    # 2. THE TRANSLATOR: Gemini translates it for the specific user
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    
    system_prompt = f"""
    You are an Official Government Policy Assistant. 
    Persona: The user is a {user_job} from {state}.
    RULE: You MUST only use the 'Search Results' provided. If the answer is not in the search results, say "I cannot find an official policy for this right now."
    Tone: Professional, simple, and direct. Use bullet points.
    """
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"LIVE DATA FROM GOV SITES: {live_gov_context}\n\nUSER QUESTION: {user_query}")
    ]
    
    response = llm.invoke(messages)
    return response.content
