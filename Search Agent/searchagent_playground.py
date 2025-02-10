from phi.agent import Agent, RunResponse 
from phi.tools.duckduckgo import DuckDuckGo
from phi.model.groq import Groq
from phi.tools.googlesearch import GoogleSearch
import phi.api
import os
from dotenv import load_dotenv

import phi
from phi.playground import Playground, serve_playground_app
import os
from dotenv import load_dotenv

load_dotenv()
PHI_API_KEY = os.getenv("PHI_API_KEY")

if not PHI_API_KEY:
    raise ValueError("PHI_API_KEY is missing! Check your .env file.")

os.environ["PHI_API_KEY"] = PHI_API_KEY  # Explicitly set it

web_search_agent = Agent(
    name='Web Search Agent',
    role="Search the web for the information",
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[DuckDuckGo()],
    instructions=["Always include sources"],
    show_tool_calls=True,
    markdown=True,

)

google_search_agent = Agent(
    name='Google Search Agent',
    role="Search the web for the information",
    description="You are a news agent that helps users find the latest news.",
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[GoogleSearch()],
    instructions=["Use tables to show data"],
    show_tool_calls=True,
    markdown=True,
)

app=Playground(agents=[google_search_agent, web_search_agent]).get_app()

if __name__ == "__main__":
    serve_playground_app("searchagent_playground:app",reload=True)