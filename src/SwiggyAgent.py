from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel as PydanticBaseModel, Field
from typing import Annotated, Sequence
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph.message import add_messages
from utilities.agent_utils import read_prompt, history_formatter, llm
from langgraph.checkpoint.memory import MemorySaver
from tools.tools import web_search_tool,entity_detector_tool
from datetime import datetime

### State model
class AgentHistory(PydanticBaseModel):
    userQuery: str
    convId: str
    messages: Annotated[Sequence[BaseMessage], add_messages] = Field(default=[])
    current_agent: str = Field(default="NONE", description="The current agent in the conversation")


### Agent
async def supervisor_agent(state:AgentHistory):
    print("------SUPERVISOR AGENT------")
    chat_history = state.messages
    query = state.userQuery
    current_agent = state.current_agent
    prompt_file = "supervisor_prompt.txt"
    system_prompt = read_prompt(prompt_file)

    # Compose the prompt (all required info in context)
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user","query:{query},current_agent:{current_agent},chat_history:{chat_history}")])
    tools = [web_search_tool,entity_detector_tool] 
    llm_with_tool = llm.bind_tools(tools)
    chain = prompt | llm_with_tool
    response = await chain.ainvoke({"query":query,"chat_history":history_formatter(chat_history), "current_agent":current_agent})
    print(response)
    if response.tool_calls:
        tool_message = AIMessage(content = '', tool_calls=response.tool_calls)
        print("processing tool call")
        return {"messages":[tool_message]}
 
    return {"messages":[response]}

def should_continue(state: AgentHistory):
    last_message = state.messages[-1]
    return 'TOOL_EXECUTOR' if getattr(last_message, 'tool_calls', None) else 'END'

async def tool_executor(state:AgentHistory):
    print("------TOOL EXECUTOR------")

    mapping = {
        "WEB_SEARCH_TOOL": web_search_tool,
        "ENTITY_DETECTOR_TOOL": entity_detector_tool
}

    last_message = state.messages[-1]
    tool_calls = last_message.tool_calls

    responses = []

    for tool_call in tool_calls:
        tool_name = tool_call['name']
        tool_args = tool_call['args']
        tool_id = tool_call['id']

        try:
            tool = mapping[tool_name]
            print(f"Executing tool: {tool_name} with args: {tool_args}")

            if tool_name == "WEB_SEARCH_TOOL":
                start_time = datetime.now()
                result = await web_search_tool.ainvoke({
                    "userQuery": tool_args.get("userQuery")
                })
                end_time = datetime.now()
                print(f"Tool executed successfully: {tool_name} in duration { (end_time - start_time).total_seconds() } seconds")

            elif tool_name == "ENTITY_DETECTOR_TOOL":
                start_time = datetime.now()
                result = await entity_detector_tool.ainvoke({
                    "userQuery": tool_args.get("userQuery"),
                    "messages": state.messages
                })
                end_time = datetime.now()
                print(f"Tool executed successfully: {tool_name} in duration { (end_time - start_time).total_seconds() } seconds")

            else:
                # Existing logic for other tools
                start_time = datetime.now()
                result = await tool.ainvoke(tool_args)
                end_time = datetime.now()
                print(f"Tool executed successfully: {tool_name} with result: {result}")
                print(f"Tool executed successfully: {tool_name} with duration: { (end_time - start_time).total_seconds() } seconds")

            responses.append(ToolMessage(content=str(result), tool_call_id=tool_id))

        except Exception as e:
            error_message = f"Error in tool '{tool_name}': {repr(e)}"
            responses.append(ToolMessage(content=error_message, tool_call_id=tool_id))


    return {
        "messages": responses
    }

workflow = StateGraph(AgentHistory)
workflow.add_node("SUPERVISOR_AGENT", supervisor_agent)
workflow.add_node("TOOL_EXECUTOR", tool_executor)
workflow.add_edge(START, "SUPERVISOR_AGENT")
workflow.add_conditional_edges("SUPERVISOR_AGENT", should_continue, {
    'TOOL_EXECUTOR': 'TOOL_EXECUTOR',
    'END': END
})
workflow.add_edge("TOOL_EXECUTOR", "SUPERVISOR_AGENT")
memory = MemorySaver()
graph = workflow.compile(checkpointer=memory)

