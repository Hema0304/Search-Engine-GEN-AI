import streamlit as st 
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import (
    ArxivQueryRun,
    WikipediaQueryRun,
    DuckDuckGoSearchRun
)
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain.callbacks import StreamlitCallbackHandler
from dotenv import load_dotenv

load_dotenv()

# ---------------- TOOLS ---------------- #
wiki = WikipediaQueryRun(
    api_wrapper=WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=300)
)

arxiv = ArxivQueryRun(
    api_wrapper=ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=300)
)

search = DuckDuckGoSearchRun(name="Search")

tools = [search, arxiv, wiki]

# ---------------- UI ---------------- #
st.title("🔍 AI Search Assistant (Groq + LangChain)")

api_key = st.sidebar.text_input("Enter Groq API Key", type="password")

# Chat memory
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! I can search the web. Ask me anything."}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# ---------------- CHAT ---------------- #
if prompt := st.chat_input("Ask anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    llm = ChatGroq(
        groq_api_key=api_key,
        model_name="qwen/qwen3-32b",
        streaming=False
    )

    # Prompt (MANDATORY in new API)
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are a smart assistant that uses tools when needed."),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}")
    ])

    # Agent
    agent = create_tool_calling_agent(llm, tools, prompt_template)

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True
    )

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container())

        response = agent_executor.invoke(
            {"input": prompt},
            {"callbacks": [st_cb]}
        )

        output = response["output"]

        st.session_state.messages.append({
            "role": "assistant",
            "content": output
        })

        st.write(output)