import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq

from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun, ArxivQueryRun

from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, create_tool_calling_agent

# -----------------------
# ENV
# -----------------------
load_dotenv()

st.set_page_config(page_title="AI Search Assistant", layout="wide")
st.title("🔎 AI Search Assistant (Groq + Tools)")

# -----------------------
# API KEY
# -----------------------
api_key = st.sidebar.text_input("Enter Groq API Key", type="password")

if not api_key:
    st.stop()

# -----------------------
# LLM
# -----------------------
llm = ChatGroq(
    groq_api_key=api_key,
    model_name="llama3-8b-8192"
)

# -----------------------
# TOOLS
# -----------------------
wiki = WikipediaQueryRun(
    api_wrapper=WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=500)
)

arxiv = ArxivQueryRun(
    api_wrapper=ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=500)
)

search = DuckDuckGoSearchRun()

tools = [search, wiki, arxiv]

# -----------------------
# PROMPT (CRITICAL)
# -----------------------
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful AI assistant. "
     "Use tools when needed: web search, wikipedia, arxiv."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])

# -----------------------
# AGENT (MODERN LCEL)
# -----------------------
agent = create_tool_calling_agent(llm, tools, prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True
)

# -----------------------
# CHAT UI
# -----------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

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
