import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq

from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun, ArxivQueryRun

from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, create_tool_calling_agent

# -----------------------------
# SETUP
# -----------------------------
load_dotenv()
st.set_page_config(page_title="AI Search Assistant")

st.title("🔎 AI Search Assistant (Groq + Tools)")

api_key = st.sidebar.text_input("Enter Groq API Key", type="password")

if not api_key:
    st.warning("Enter Groq API Key")
    st.stop()

# -----------------------------
# TOOLS
# -----------------------------
wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
arxiv = ArxivQueryRun(api_wrapper=ArxivAPIWrapper())
search = DuckDuckGoSearchRun()

tools = [search, wiki, arxiv]

# -----------------------------
# LLM
# -----------------------------
llm = ChatGroq(
    groq_api_key=api_key,
    model_name="llama3-8b-8192"
)

# -----------------------------
# PROMPT
# -----------------------------
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a smart AI assistant. "
     "Use tools when needed: web search, wikipedia, arxiv."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])

# -----------------------------
# AGENT (MODERN WAY)
# -----------------------------
agent = create_tool_calling_agent(llm, tools, prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True
)

# -----------------------------
# CHAT MEMORY
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! Ask me anything."}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# -----------------------------
# INPUT
# -----------------------------
query = st.chat_input("Ask anything...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    st.chat_message("user").write(query)

    with st.chat_message("assistant"):
        result = agent_executor.invoke({"input": query})
        answer = result["output"]

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )

        st.write(answer)
