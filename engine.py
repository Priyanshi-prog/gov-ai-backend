import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import SystemMessage, HumanMessage

def get_ai_recommendation(user_job, state, user_query, age, gender, income):
    # 1. LIVE SEARCH: Finds everything related to the topic
    search = TavilySearchResults(max_results=3, include_domains=["gov.in", "nic.in"])
    search_query = f"government schemes for {user_job} in {state} {user_query} 2026"
    live_gov_context = search.run(search_query)

    # 2. PERSONALIZED FILTER: Gemini acts as a screening officer
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
    
    system_prompt = f"""
    You are an Official Government Policy Eligibility Expert.
    
    USER PROFILE:
    - Occupation: {user_job}
    - Location: {state}
    - Age: {age}
    - Gender: {gender}
    - Annual Income: {income}

    TASK: 
    1. Read the provided search results.
    2. Suggest ONLY the schemes where the user matches the eligibility criteria based on their age, gender, and income.
    3. If a scheme is for 'Women only' and the user is 'Male', ignore it.
    4. If a scheme is for 'BPL/Low Income' and the user earns 10 Lakhs, ignore it.
    5. Be very specific about WHY they qualify.
    """
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"LIVE GOV DATA: {live_gov_context}\n\nUSER REQUEST: {user_query}")
    ]
    
    response = llm.invoke(messages)
    return response.content
