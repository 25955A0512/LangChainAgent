# 🤖 AI Agent using LangChain + Groq

An AI-powered conversational agent built using LangChain, Groq Llama 3.3, Streamlit, and external tools for real-time reasoning and information retrieval.

---

## 🚀 Features

- 🔎 Real-time Web Search using DuckDuckGo
- 🌐 Tavily Search Integration
- 📚 Wikipedia Knowledge Retrieval
- ➗ Mathematical Calculations using Math Tool
- 💬 Interactive Chat Interface with Streamlit
- ⚡ Powered by Groq Llama 3.3 Model
- 🧠 Tool-based AI Agent Workflow using LangChain

---

## 🛠️ Technologies Used

- Python
- Streamlit
- LangChain
- Groq API
- DuckDuckGo Search
- Tavily Search API
- Wikipedia Tool
- LLM Math Tool

---

## 📂 Project Structure

```bash
.
├── App.py
├── requirements.txt
├── .env
└── README.md
Install Dependencies
pip install -r requirements.txt
🔑 Environment Variables

Create a .env file in the root directory.

GROQ_API_KEY=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key
▶️ Run the Application
streamlit run App.py
🧠 How It Works

The AI Agent uses LangChain Agents with integrated external tools.

Workflow:
User enters a query
Agent decides which tool to use
Tools retrieve information
Groq LLM processes the results
Final response is generated

🔍 Integrated Tools
Tool	Purpose
DuckDuckGo	Real-time web search
Tavily Search	Advanced AI search retrieval
Wikipedia	Knowledge lookup
Math Tool	Mathematical reasoning

📚 Learning Outcomes
Through this project, I learned:
LangChain Agents
Tool Calling
Prompt Engineering
LLM Orchestration
AI Workflow Design
Generative AI Application Development

🔮 Future Improvements
Add Conversation Memory
Integrate RAG Pipelines
Add Multi-Agent Workflows
Voice-based Interaction
Deploy on Cloud Platforms

👨‍💻 Author
Laxman Nayak

⭐ If you like this project

Give it a star on GitHub ⭐
