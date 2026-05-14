import warnings
warnings.filterwarnings("ignore")

import os
import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq

from langchain.agents.initialize import initialize_agent
from langchain.agents.agent_types import AgentType

from langchain_community.agent_toolkits.load_tools import load_tools

from langchain_core.tools import Tool

from duckduckgo_search import DDGS

import requests

# Load Environment Variables
load_dotenv()

# ---------------------------------------------------
# Streamlit Config
# ---------------------------------------------------

st.set_page_config(
    page_title="AI Agent",
    page_icon="🤖"
)

st.title("🤖 AI Agent")

st.caption("Powered by Groq + DuckDuckGo + Tavily + Math")

# ---------------------------------------------------
# Sidebar
# ---------------------------------------------------

with st.sidebar:

    st.header("🔑 API Keys")

    groq_key = st.text_input(
        "Groq API Key",
        type="password",
        placeholder="gsk_..."
    )

    tavily_key = st.text_input(
        "Tavily API Key",
        type="password",
        placeholder="tvly-..."
    )

    st.markdown("---")

    st.caption(
        "Get free keys at:\n\n"
        "console.groq.com\n\n"
        "app.tavily.com"
    )

# ---------------------------------------------------
# Require Groq Key
# ---------------------------------------------------

if not groq_key:

    st.info("👈 Enter your Groq API Key in the sidebar.")

    st.stop()

# ---------------------------------------------------
# Load Agent
# ---------------------------------------------------

@st.cache_resource
def load_agent(groq_api_key, tavily_api_key):

    # LLM
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=groq_api_key,
        temperature=0
    )

    # ------------------------------------------------
    # Search Function
    # ------------------------------------------------

    def search(query):

        # Tavily Search
        if tavily_api_key:

            try:

                response = requests.post(
                    "https://api.tavily.com/search",
                    json={
                        "api_key": tavily_api_key,
                        "query": query,
                        "max_results": 5
                    },
                    timeout=10
                )

                data = response.json()

                results = data.get("results", [])

                if results:

                    return "\n\n".join(
                        f"{r.get('title', '')}\n{r.get('content', '')}"
                        for r in results
                    )

            except Exception:
                pass

        # DuckDuckGo Fallback
        try:

            with DDGS() as ddgs:

                results = list(
                    ddgs.text(query, max_results=5)
                )

            if results:

                return "\n\n".join(
                    f"{r.get('title', '')}\n{r.get('body', '')}"
                    for r in results
                )

        except Exception:
            pass

        return "No results found."

    # ------------------------------------------------
    # Tools
    # ------------------------------------------------

    search_tool = Tool(
        name="Search",
        func=search,
        description=(
            "Search the internet for current information, "
            "news, people, facts, and events."
        )
    )

    math_tool = load_tools(
        ["llm-math"],
        llm=llm
    )[0]

    tools = [
        search_tool,
        math_tool
    ]

    # ------------------------------------------------
    # Agent
    # ------------------------------------------------

    agent = initialize_agent(
        tools=tools,
        llm=llm,

        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,

        verbose=False,

        handle_parsing_errors=True,

        max_iterations=5
    )

    return agent

# Initialize Agent
agent_executor = load_agent(groq_key, tavily_key)

# ---------------------------------------------------
# Chat History
# ---------------------------------------------------

if "messages" not in st.session_state:

    st.session_state.messages = []

# Display Previous Messages
for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):

        st.markdown(msg["content"])

# ---------------------------------------------------
# Chat Input
# ---------------------------------------------------

if prompt := st.chat_input("Ask me anything..."):

    # Save User Message
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    # Display User Message
    with st.chat_message("user"):

        st.markdown(prompt)

    # Assistant Response
    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            try:

                response = agent_executor.run(prompt)

            except Exception as e:

                response = f"❌ Error: {str(e)}"

        st.markdown(response)

    # Save Assistant Response
    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })
