# ============================================================
# AI Search Assistant — Web + PubMed + Wikipedia
# Tools: DuckDuckGo (Web) + PubMed (Research Papers) + Wikipedia
# Great for: medical, scientific, and general knowledge questions
# Requirements:
#   pip install streamlit langchain langchain-groq
#               langchain-community wikipedia duckduckgo-search
#               xmltodict  (PubMed dependency)
# ============================================================

import streamlit as st
import wikipedia
from langchain_groq import ChatGroq
<<<<<<< Updated upstream

from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper  #Wrap external APIs into usable format
from langchain_community.tools import (
    ArxivQueryRun,
    WikipediaQueryRun,
    DuckDuckGoSearchRun
)


#WikipediaQueryRun → fetch Wikipedia info
# ArxivQueryRun → fetch research papers
# DuckDuckGoSearchRun → web search


from dotenv import load_dotenv
import os

load_dotenv()

from langchain_groq import ChatGroq



llm = ChatGroq(
    api_key=st.secrets["GROQ_API_KEY"],
    model="llama3-8b-8192"
)

=======
from langchain_community.utilities import WikipediaAPIWrapper, PubMedAPIWrapper
from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.tools import Tool                          # Fix: wrap PubMed in plain Tool
>>>>>>> Stashed changes
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain.callbacks import StreamlitCallbackHandler

# ── Fix: Wikipedia blocks requests without a proper User-Agent ───────────────
wikipedia.set_user_agent("AISearchAssistant/1.0 (contact@example.com)")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Search Assistant",

<<<<<<< Updated upstream
wiki = WikipediaQueryRun(
    api_wrapper=WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=300)
=======
    layout="centered"
>>>>>>> Stashed changes
)

st.title(" AI Search Assistant")


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header(" Configuration")
api_key = st.sidebar.text_input("Groq API Key", type="password", placeholder="gsk_...")
st.sidebar.markdown("---")
st.sidebar.markdown("**Active Tools**")
st.sidebar.markdown(" DuckDuckGo — general web search")
st.sidebar.markdown(" PubMed — medical & science research papers")
st.sidebar.markdown(" Wikipedia — encyclopedic knowledge")
st.sidebar.markdown("---")
st.sidebar.markdown("Get a free key at [console.groq.com](https://console.groq.com)")

# ── Tools setup ───────────────────────────────────────────────────────────────

# Tool 1: DuckDuckGo — general web search, no API key needed
search_tool = DuckDuckGoSearchRun(
    name="DuckDuckGo_Search",
    description=(
        "Search the web for current events, news, general information, "
        "and any topic not covered by other tools."
    )
)

# Tool 2: PubMed — wrapped in plain Tool to fix CallbackManagerForToolRun
# serialization error that occurs with PubmedQueryRun directly
_pubmed_wrapper = PubMedAPIWrapper(
    top_k_results=2,            # fetch top 2 papers
    doc_content_chars_max=500   # keep responses concise
)

pubmed_tool = Tool(
    name="PubMed_Search",
    func=lambda query: _pubmed_wrapper.run(query),   # plain callable, no run_manager passed
    description=(
        "Search PubMed for peer-reviewed medical and scientific research papers. "
        "Use this for health, biology, medicine, clinical trials, and science topics."
    )
)

<<<<<<< Updated upstream

st.title("Hybrid Semantic Search Engine for Document Retrieval")
=======
# Tool 3: Wikipedia — encyclopedic reference
wiki_tool = WikipediaQueryRun(
    api_wrapper=WikipediaAPIWrapper(
        top_k_results=1,
        doc_content_chars_max=500
    ),
    description=(
        "Search Wikipedia for encyclopedic information about people, places, "
        "history, science, and general concepts."
    )
)
>>>>>>> Stashed changes

tools = [search_tool, pubmed_tool, wiki_tool]

# ── Chat memory ────────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
            
                "Ask me anything!"
            )
        }
    ]

# Display chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

<<<<<<< Updated upstream

=======
# ── Handle user input ─────────────────────────────────────────────────────────
>>>>>>> Stashed changes
if prompt := st.chat_input("Ask anything..."):

    if not api_key:
        st.warning(" Please enter your Groq API key in the sidebar.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # ── LLM ──────────────────────────────────────────────────────────────────
    llm = ChatGroq(
        groq_api_key=api_key,
        model_name="qwen/qwen3-32b",
        streaming=False
    )

    # ── Prompt template ───────────────────────────────────────────────────────
    prompt_template = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a smart AI search assistant with access to three tools:
1. DuckDuckGo_Search — for current events, news, and general web information
2. PubMed_Search     — for peer-reviewed medical and scientific research papers
3. wikipedia         — for encyclopedic facts about people, places, history, and science

Always choose the most appropriate tool for the question:
- Medical / health / biology / clinical question → use PubMed_Search first
- Historical / factual / concept topic           → use wikipedia first
- Current events or anything else                → use DuckDuckGo_Search

Always cite your sources clearly in the final answer."""
        ),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}")
    ])

    # ── Agent ─────────────────────────────────────────────────────────────────
    agent = create_tool_calling_agent(llm, tools, prompt_template)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,   # gracefully handle malformed LLM output
        max_iterations=5              # prevent infinite loops
    )

    # ── Run and stream response ───────────────────────────────────────────────
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=True)
        try:
            response = agent_executor.invoke(
                {"input": prompt},
                {"callbacks": [st_cb]}
            )
            output = response["output"]
        except Exception as e:
            output = (
                f" Something went wrong: {str(e)}\n\n"
                "Please try again or rephrase your question."
            )

<<<<<<< Updated upstream
        st.write(output)
=======
        st.session_state.messages.append({"role": "assistant", "content": output})
        st.write(output)
>>>>>>> Stashed changes
