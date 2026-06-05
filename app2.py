import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq

from langchain_community.utilities import (
    WikipediaAPIWrapper,
    ArxivAPIWrapper
)

from langchain_community.tools import (
    WikipediaQueryRun,
    ArxivQueryRun,
    DuckDuckGoSearchRun
)

from langchain.agents import (
    AgentExecutor,
    create_tool_calling_agent
)

from langchain_core.prompts import ChatPromptTemplate
from langchain.callbacks import StreamlitCallbackHandler

# -------------------------
# Load Environment Variables
# -------------------------
load_dotenv()

# -------------------------
# Wikipedia Tool
# -------------------------
wiki = WikipediaQueryRun(
    api_wrapper=WikipediaAPIWrapper(
        top_k_results=1,
        doc_content_chars_max=500
    )
)

# -------------------------
# Arxiv Tool
# -------------------------
arxiv = ArxivQueryRun(
    api_wrapper=ArxivAPIWrapper(
        top_k_results=3,
        doc_content_chars_max=1000
    )
)

# -------------------------
# DuckDuckGo Search Tool
# -------------------------
search = DuckDuckGoSearchRun(name="Search")

# -------------------------
# Tool List
# -------------------------
tools = [
    search,
    wiki,
    arxiv
]

# -------------------------
# Streamlit UI
# -------------------------
st.title("AI Search Assistant")
st.caption("Web Search + Wikipedia + Arxiv Research Papers")

api_key = st.sidebar.text_input(
    "Enter Groq API Key",
    type="password"
)

if not api_key:
    st.info("Please enter your Groq API key.")
    st.stop()

# -------------------------
# Chat Memory
# -------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hi! I can search the web, Wikipedia, and Arxiv research papers."
        }
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# -------------------------
# User Input
# -------------------------
if prompt := st.chat_input("Ask anything..."):

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    st.chat_message("user").write(prompt)

    # -------------------------
    # LLM
    # -------------------------
    llm = ChatGroq(
        groq_api_key=api_key,
        model_name="qwen/qwen3-32b",
        streaming=False
    )

    # -------------------------
    # Prompt
    # -------------------------
    prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are a smart AI assistant.

You have access to:

1. Search
   - Use for current events, news, websites, and general web information.

2. Wikipedia
   - Use for encyclopedic facts, people, places, history, concepts.

3. Arxiv
   - Use for research papers, academic topics, machine learning, AI, deep learning,
     healthcare prediction, scientific research, algorithms, and technical studies.

Always choose the most appropriate tool.
For research-related questions, prefer Arxiv.
"""
            ),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}")
        ]
    )

    # -------------------------
    # Agent
    # -------------------------
    agent = create_tool_calling_agent(
        llm,
        tools,
        prompt_template
    )

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True
    )

    # -------------------------
    # Response
    # -------------------------
    with st.chat_message("assistant"):

        st_cb = StreamlitCallbackHandler(st.container())

        try:

            response = agent_executor.invoke(
                {"input": prompt},
                config={"callbacks": [st_cb]}
            )

            output = response["output"]

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": output
                }
            )

            st.write(output)

        except Exception as e:

            st.error(f"Error: {str(e)}")
