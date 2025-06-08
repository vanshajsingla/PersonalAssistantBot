from fastapi import FastAPI, HTTPException
import uvicorn
import pickle
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing import List
import os
from typing import Annotated, Sequence
from langchain_core.messages import BaseMessage
from langgraph.pregel.types import StateSnapshot
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from SwiggyAgent import graph
BASE_TRANSIENT_DIR = "fileshare/transients"

app = FastAPI()
app.title = "Swiggy Assistant Agent API"

class ConversationMessage(BaseModel):
    responseType: str = Field(..., title="Type of the response (e.g., 'text', 'Json')")
    responseData: str = Field(..., title="Response data, such as json content or plain text")

class swiggyAgentResponseModel(BaseModel):
    status: int = Field(..., title="Status of the response")
    conversationId: str = Field(..., title="Unique identifier for the conversation")
    conversationState: str = Field(default="beginning", title="Current state of the conversation")
    responseMessage : str = Field(..., title="Response message")
    conversationMessages: List[ConversationMessage] = Field(
        None, title="Messages from the agent"
    )

class swiggyAgentRequestModel(BaseModel):
    conversationId: str = Field(title="Conversation ID", description="Unique identifier for the conversation")
    conversationState: str = Field(default="beginning", title="Conversation State", description="State of the conversation, e.g., 'beginning', 'ongoing', 'ended'")
    userInput: str = Field(title="User Input", description="Input provided by the user")


class AgentHistory(BaseModel):
    userQuery: str
    convId: str
    messages: Annotated[Sequence[BaseMessage], add_messages] = Field(default=[])
    current_agent: str = Field(default="NONE", description="The current agent in the conversation")


def get_state_file_path(conv_id: str) -> str:
    """
    Returns the file path for the AgentHistory pickle file in fileshare/transients/<conv_id>/state_obj.pkl
    """
    transient_store = os.path.join(BASE_TRANSIENT_DIR, conv_id)
    # Ensure directory exists
    os.makedirs(transient_store, exist_ok=True)
    state_file = os.path.join(transient_store, "state_obj.pkl")
    return os.path.normpath(state_file)

def load_state(file_path: str):
    """
    Load the AgentHistory state from the given file path.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"State file does not exist: {file_path}")
    with open(file_path, 'rb') as f:
        state = pickle.load(f)
    return state

def save_state(state, file_path: str):
    """
    Save the AgentHistory state to the given file path.
    """
    # Ensure the directory exists
    directory = os.path.dirname(file_path)
    os.makedirs(directory, exist_ok=True)
    with open(file_path, 'wb') as f:
        pickle.dump(state, f)

def get_final_response(current_agent, state):
    """
    Determines the final response based on the current agent and state.
    """
    if current_agent in ["NONE", "SUPERVISOR_AGENT"]:
        for message in reversed(state.values['messages']):
            if isinstance(message, AIMessage):
                return message.content
        return "Please provide more details. I'm here to help!"
    elif current_agent == "END":
        return "Thanks for using the Swiggy Agent. Have a great day!"
    else:
        return "Please provide more details. I'm here to help!"

@app.post('/SwiggyAssistantAgent', response_model=swiggyAgentResponseModel)
async def swiggyAgent(
        body: swiggyAgentRequestModel,
    ):
    try:
        conversation_id = body.conversationId
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    
    try:
        user_input = body.userInput
        # Retrieve or initialize conversation state
        state_file_path = get_state_file_path(conversation_id)

        if os.path.exists(state_file_path):
            state = load_state(state_file_path)
        else:
            state = AgentHistory(userQuery="", convId=conversation_id)

        thread_config = {"configurable": {"thread_id": conversation_id}}

        if isinstance(state, StateSnapshot):
            print("-------State object is a snapshot-------")
            new_values = dict(state.values)
            new_values["userQuery"] = user_input
            new_values["messages"] = new_values.get("messages", []) + [HumanMessage(content=user_input)]
            new_state = AgentHistory.model_validate(new_values)
            response = await graph.ainvoke(new_state, thread_config)
            print(f"Invoked workflow graph | Response: {response}")
        else:
            print("-------State object is not a snapshot-------")
            state.userQuery = user_input
            state.messages.append(HumanMessage(content=user_input))
            response = await graph.ainvoke(state, config=thread_config)
            print(f"Invoked workflow graph | Response: {response}")


        state = graph.get_state(thread_config)
        save_state(state, state_file_path)
        print(f"Saved updated state | State File Path: {state_file_path} | Updated State: {state}")

        # Determine the final response based on current_agent
        current_agent = state.values["current_agent"]
        final_response = get_final_response(current_agent, state)
        
        # Construct the conversation message
        conversation_messages = [
            ConversationMessage(
                responseType="Json",
                responseData=final_response
            )
        ]

        # Construct the response model
        response_object = swiggyAgentResponseModel(
            status=1,
            conversationId=conversation_id,
            conversationState="ended" if current_agent == "END" else "ongoing",
            responseMessage="Answer generated successfully",
            conversationMessages=conversation_messages
        )

    except Exception as exception:
        print(f"Error processing request | Error: {str(exception)}")
        response_object = swiggyAgentResponseModel(
            status=0,
            conversationId=body.conversationId,
            conversationState="ongoing",
            responseMessage="Internal server error.",
            conversationMessages=[
                ConversationMessage(
                    responseType="text",
                    responseData="Internal server error."
                )
            ]
        )

    return response_object

def main():
    """ main """
    # run the fastapi app
    uvicorn.run('main:app', host='0.0.0.0', port=4005, workers=1, log_level='debug', reload=True)


if __name__ == '__main__':

    main()
