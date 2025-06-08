import streamlit as st
import uuid
import requests
from langchain_core.messages import HumanMessage
import json, re

st.set_page_config(page_title="Personal Assistant Bot", layout="wide")
st.title("Personal Assistant Bot")

# ---- Session State Management ----
st.sidebar.header("Session Info")
if 'convId' not in st.session_state:
    st.session_state.convId = f"Assistant_{str(uuid.uuid4())[:8]}"
if 'conversationState' not in st.session_state:
    st.session_state.conversationState = "beginning"
if 'history' not in st.session_state:
    st.session_state.history = []

st.sidebar.write(f"Conversation ID: {st.session_state.convId}")
st.sidebar.write(f"Conversation State: {st.session_state.conversationState}")

# ---- API URL ----
API_URL = "http://localhost:4005/SwiggyAssistantAgent"

# ---- Show Chat Messages ----
def display_chat_messages():
    for msg in st.session_state.history:
        if isinstance(msg, HumanMessage):
            st.chat_message("user").write(msg.content)
        else:
            st.json(msg)

# ---- API Call ----
def call_agent_api(user_query):
    headers = {
        "Content-Type": "application/json"
    }

    payload = {
        "conversationId": st.session_state.convId,
        "conversationState": st.session_state.conversationState,
        "userInput": user_query
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()

        if data.get("conversationMessages"):
            msg = data["conversationMessages"][0]
            # Try to parse as JSON if possible
            try:
                json_obj = json.loads(re.sub(r'^```[a-zA-Z]*\n|```$', '', msg["responseData"].strip()))
                
                return json_obj, data.get("conversationState", "ongoing")
            except Exception:
                # Not JSON, just text fallback
                return msg["responseData"], data.get("conversationState", "ongoing")
        else:
            st.warning("No valid conversation message received.")
            return None, data.get("conversationState", "ongoing")
    except requests.RequestException as e:
        st.error(f"API request failed: {e}")
        return None, "ongoing"

# ---- Main Chat UI ----
st.subheader("Chat")
display_chat_messages()

user_input = st.chat_input("Type your request...")

if user_input:
    st.session_state.history.append(HumanMessage(content=user_input))
    st.chat_message("user").write(user_input)

    api_response, new_state = call_agent_api(user_input)

    if api_response:
        st.session_state.history.append(api_response)
        if isinstance(api_response, dict):
            st.json(api_response)
        else:
            st.markdown(api_response, unsafe_allow_html=True)
    st.session_state.conversationState = new_state

# ---- Reset Button ----
if st.sidebar.button("Reset Conversation"):
    st.session_state.history = []
    st.session_state.convId = f"Assistant_{str(uuid.uuid4())[:8]}"
    st.session_state.conversationState = "beginning"
    st.rerun()
