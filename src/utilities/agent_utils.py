from dotenv import load_dotenv
import os
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage

load_dotenv()

# Access environment variables
azure_endpoint = os.getenv("AZURE_ENDPOINT")
azure_deployment = os.getenv("AZURE_DEPLOYMENT")
api_key = os.getenv("AZURE_API_KEY")
api_version = os.getenv("AZURE_API_VERSION")

# Initialize AzureChatOpenAI using environment variables
llm = AzureChatOpenAI(
    azure_endpoint=azure_endpoint,
    azure_deployment=azure_deployment,
    api_key=api_key,
    api_version=api_version,
    temperature=0.0001
)

BASE_PATH = r"fileshare\prompts"
def read_prompt(file_name):
    file_path = os.path.join(BASE_PATH, file_name).replace("\\", "/")
    with open(file_path, 'r') as file:
        prompt = file.read()
    
    return prompt

def history_formatter(messages):
    chat_list = []

    for i in messages:
        msg_content = i.content if hasattr(i, 'content') else ''
        
        if isinstance(i, HumanMessage):
            chat_list.append(f"Human: {msg_content}")
        
        # AIMessage check with/without tool calls
        elif isinstance(i, AIMessage):
            if i.tool_calls:
                tool_call = i.tool_calls[0]  # Assuming only the first tool call is needed
                chat_list.append(f"AI (with tool calls): tool_name: '{tool_call['name']}', tool arguments: '{tool_call['args']}', call id: '{tool_call['id']}'")
            else:
                chat_list.append(f"AI: '{msg_content}'")
        
        elif isinstance(i, ToolMessage):
            chat_list.append(f"Tool call id: '{i.tool_call_id}', Tool Response: '{msg_content}'")
        
        elif isinstance(i, SystemMessage):
            chat_list.append(f"System: {msg_content}")
    
    # Join the list of chat strings with newline characters
    return "\n".join(chat_list)

