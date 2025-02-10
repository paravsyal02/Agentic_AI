from phi.agent import Agent, RunResponse 
from phi.tools.duckduckgo import DuckDuckGo
from phi.model.groq import Groq
from phi.tools.googlesearch import GoogleSearch

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
    instructions=["Use tables to show data" "Use tables to show data"],
    show_tool_calls=True,
    markdown=True,
)

multi_ai_agent = Agent(
    team = [web_search_agent, google_search_agent],
    model=Groq(id="llama-3.3-70b-versatile"),
    instructions=["always include resources", "Use tables to show data"],
    show_tool_calls=True,
    markdown=True,
)   

multi_ai_agent.print_response(" Latest news about Deep Seek AI", markdown=True)