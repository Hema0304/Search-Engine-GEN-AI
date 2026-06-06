import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import Tool
from langchain.agents import initialize_agent, AgentType

# -----------------------
# ENV
# -----------------------
load_dotenv()

st.set_page_config(page_title="AI Search Assistant", layout="wide")
st.title("🔎 AI Search Assistant (Groq + Stable Tools)")

# -----------------------
# API KEY
# -----------------------
api_key = st.sidebar.text_input("Enter Groq API Key", type="password")

if not api_key:
    st.stop()

api_key = api_key.strip()

# -----------------------
# LLM (STABLE GROQ MODEL)
# -----------------------
llm = ChatGroq(
    groq_api_key=api_key,
    model_name="llama-3.1-8b-instant",
    temperature=0.2,
    max_tokens=1024
)

# -----------------------
# TOOL (ONLY STABLE ONE)
# -----------------------
search = DuckDuckGoSearchRun()

tools = [
    Tool(
        name="WebSearch",
        func=search.run,
        description="Use this tool for real-time web search and general knowledge questions."
    )
]

# -----------------------
# AGENT (SAFE CONFIG)
# -----------------------
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    handle_parsing_errors=True
)

# -----------------------
# SESSION STATE
# -----------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# -----------------------
# USER INPUT
# -----------------------
query = st.chat_input("Ask anything...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    st.chat_message("user").write(query)

    with st.chat_message("assistant"):
        try:
            result = agent.invoke({"input": query})
            answer = result.get("output", "No response generated.")
        except Exception as e:
            answer = f"Error occurred: {str(e)}"

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )

        st.write(answer)
