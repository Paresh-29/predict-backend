import os
from dotenv import load_dotenv
from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.yfinance import YFinanceTools
from app.utils.text_utils import clean_markdown  


load_dotenv()


groq_api_key = os.environ["GROQ_API_KEY"]

# News Agent
news_agent = Agent(
    name="News Agent",
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[DuckDuckGo()],
     instructions=[
         "Fetch and summarize the latest financial news for the company.",
         "Highlight points affecting stock performance.",
         "Include credible sources."
     ]
)

# Financial Data Analyzer
finance_agent = Agent(
    name="Finance Agent",
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[
        YFinanceTools(
            stock_price=True,
            analyst_recommendations=True,
            stock_fundamentals=True,
        )
    ],
     instructions=[
         "Retrieve key financial metrics, including P/E ratio, market cap, dividend yield, beta, and alpha.",
         "Use tables for clarity.",
         "Focus on how these metrics impact stock performance."
 ]
)

# Aggregator Agent
aggregator_agent = Agent(
    model=Groq(id="llama-3.3-70b-versatile"),
    agents=[news_agent, finance_agent],
   instructions=[
         "Combine inputs from all agents into a concise summary.",
         "Highlight critical news and key financial metrics.",
         "Provide investment recommendations (long-term and short-term)."
         "Present all comparisons and financial data markdown table format. Do not use plain text for tables.",
        "If possible, use clear section headers and keep formatting consistent."
     ]
)


def query_aggregator_agent(stock_name: str) -> str:
    # Run the agent to get the response
    response = aggregator_agent.run(message=f"Analyze {stock_name} stock.")

    # Check if response is a dictionary with 'content' or has a 'content' attribute
    if isinstance(response, dict) and "content" in response:
        # Clean the markdown content before returning it
        return clean_markdown(response["content"])  # Clean the content
    elif hasattr(response, "content"):
        # Clean the markdown content before returning it
        return clean_markdown(response.content)  # Clean the content
    else:
        raise ValueError("Failed to get response content.")
    
     # Convert markdown to HTML (optional)
    # html_content = markdown.markdown(content, extensions=['tables'])
    # return html_content