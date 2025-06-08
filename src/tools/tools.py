from langchain_core.prompts import ChatPromptTemplate
import re
from langchain_core.tools import tool
from duckduckgo_search import DDGS
from utilities.agent_utils import read_prompt, llm
from langgraph.prebuilt import InjectedState
from typing import Annotated

@tool("WEB_SEARCH_TOOL")
async def web_search_tool(userQuery: str):
    """
    Searches DuckDuckGo for the user query and returns top 3-5 results.
    """
    with DDGS() as ddgs:
        results = [
            {
                "title": r.get("title"),
                "href": r.get("href"),
                "snippet": r.get("body")
            }
            for r in ddgs.text(userQuery, max_results=5)
        ]
    return {"results": results[:5]}


@tool("ENTITY_DETECTOR_TOOL")
def entity_detector_tool(userQuery,messages: Annotated[list, InjectedState("messages")]):
    """
    Detects entities in the user query and returns them in a structured JSON format.
    """
    assistant = read_prompt("entity_detector_prompt.txt")
    print(f"Json Prompt: {assistant}")
    prompt = ChatPromptTemplate.from_messages([
            ("system", assistant),
            ("user", "content:{content},")
        ])
    
    chain = prompt| llm
    response = chain.invoke({"content":f"User Query: {userQuery}, Chat History: {messages}"})
    response = response.content.replace("\n","")
    return response