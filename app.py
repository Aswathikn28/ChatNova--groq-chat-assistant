import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import json
import os

# Setup model
llm = ChatGroq(
    temperature=0.7,
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.3-70b-versatile"
)

# Prompt setup
prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="You are a helpful Python learning assistant."),
    MessagesPlaceholder(variable_name="history"),
    HumanMessage(content="{input}")
])

chain = prompt | llm
message_history = ChatMessageHistory()

chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: message_history,
    input_messages_key="input",
    history_messages_key="history"
)

# Load user data
try:
    with open("user_data.json", "r") as f:
        user_data = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    user_data = {"questions_asked": 0, "topics_covered": []}

# Streamlit UI
st.set_page_config(page_title="Python Learning Assistant", page_icon="🐍")
st.title("🐍 Python Learning Assistant")

if "history" not in st.session_state:
    st.session_state.history = []

# Chat input
user_input = st.chat_input("Ask me about Python...")

if user_input:
    st.session_state.history.append(("You", user_input))
    message_history.add_user_message(user_input)

    response = chain_with_history.invoke(
        {"input": user_input},
        config={"configurable": {"session_id": "user_session"}}
    )

    message_history.add_ai_message(response.content)
    st.session_state.history.append(("Assistant", response.content))

    # Update user stats
    user_data["questions_asked"] += 1
    if "python" in user_input.lower():
        if "python" not in user_data["topics_covered"]:
            user_data["topics_covered"].append("python")

    with open("user_data.json", "w") as f:
        json.dump(user_data, f)

# Display chat history
for sender, msg in st.session_state.history:
    if sender == "You":
        st.markdown(f"🧑‍💻 **You:** {msg}")
    else:
        st.markdown(f"🤖 **Assistant:** {msg}")

# Sidebar for stats
st.sidebar.header("📊 Progress Tracker")
st.sidebar.write(f"**Questions Asked:** {user_data['questions_asked']}")
st.sidebar.write(f"**Topics Covered:** {', '.join(user_data['topics_covered']) or 'None'}")
