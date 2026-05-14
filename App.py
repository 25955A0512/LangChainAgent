import warnings
warnings.filterwarnings("ignore")

import os
import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_core.tools import Tool
from langchain_core.prompts import PromptTemplate
from duckduckgo_search import DDGS
import requests

load_dotenv()

st.set_page_config(page_title="AI Agent", page_icon="🤖")
st.title("🤖 AI Agent")
st.caption("Powered by Groq + DuckDuckGo + Tavily + Math")

# ── Sidebar: API Keys ─────────────────────────────────────────
with st.sidebar:
    st.header("🔑 API Keys")
    groq_key = st.text_input("Groq API Key", type="password", placeholder="gsk_...")
    tavily_key = st.text_input("Tavily API Key", type="password", placeholder="tvly-...")
    st.markdown("---")
    st.caption("Get free keys at:\n\n[console.groq.com](https://console.groq.com)\n\n[app.tavily.com](https://app.tavily.com)")

# ── Gate: require keys ────────────────────────────────────────
if not groq_key:
    st.info("👈 Enter your **Groq API Key** in the sidebar to get started.")
    st.stop()

# ── Load Agent ────────────────────────────────────────────────
@st.cache_resource
def load_agent(groq_api_key: str, tavily_api_key: str):

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=groq_api_key,
        temperature=0
    )

    # Search Tool (Tavily → DuckDuckGo fallback)
    def search(query: str) -> str:
        if tavily_api_key:
            try:
                resp = requests.post(
                    "https://api.tavily.com/search",
                    json={"api_key": tavily_api_key, "query": query, "max_results": 5},
                    timeout=10
                )
                data = resp.json()
                results = data.get("results", [])
                if results:
                    return "\n\n".join(
                        f"{r.get('title', '')}\n{r.get('content', '')}"
                        for r in results
                    )
            except Exception:
                pass

        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=5))
            if results:
                return "\n\n".join(
                    f"{r.get('title', '')}\n{r.get('body', '')}"
                    for r in results
                )
        except Exception:
            pass

        return "No results found."

    search_tool = Tool(
        name="Search",
        func=search,
        description=(
            "Search the internet for real-time information, current events, "
            "news, people, prices, or anything factual. Input: a search query."
        )
    )

    math_tool = load_tools(["llm-math"], llm=llm)[0]

    tools = [search_tool, math_tool]

    prompt = PromptTemplate.from_template("""Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}""")

    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)

    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        handle_parsing_errors=True,
        max_iterations=8
    )

    return executor

agent_executor = load_agent(groq_key, tavily_key)

# ── Chat History ──────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Chat Input ────────────────────────────────────────────────
if prompt := st.chat_input("Ask me anything..."):

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = agent_executor.invoke({"input": prompt})
                response = result["output"]
            except Exception as e:
                response = f"Error: {str(e)}"
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
