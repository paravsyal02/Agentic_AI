from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo

import phi.api

import os
from dotenv import load_dotenv

from phi.model.groq import Groq
import phi
from phi.playground import Playground, serve_playground_app
load_dotenv()

phi.api=os.getenv("PHI_API_KEY")

# Web search agent
websearch_agent =Agent(
    name="Web Search Agent",
    role="Search the web for thr information",
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[DuckDuckGo()],
    instructions=["Always include sources"],
    show_tools_calls=True,
    markdown=True,

)


# Financial agent
finance_agent = Agent(
    name="Finance AI Agent",
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True, company_news=True)],
    instructions=["Use Tables to display the data"],
    show_tools_calls=True,
    markdown=True,

)


app=Playground(agents=[finance_agent, websearch_agent]).get_app()

if __name__ == "__main__":
    serve_playground_app("playground:app",reload=True)