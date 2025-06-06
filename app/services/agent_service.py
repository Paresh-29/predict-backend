import os
from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.yfinance import YFinanceTools
from app.utils.text_utils import clean_markdown  # Import the clean_markdown function

# Directly set the API key for testing purposes
os.environ["GROQ_API_KEY"] = "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"

# News Agent
news_agent = Agent(
    name="News Agent",
    model=Groq(id="meta-llama/llama-4-scout-17b-16e-instruct"),
    tools=[DuckDuckGo()],
    instructions=[
        "Fetch and summarize the latest financial news.",
        "Highlight points affecting stock performance.",
        "Include credible sources.",
    ],
)

# Financial Data Analyzer
finance_agent = Agent(
    name="Finance Agent",
    model=Groq(id="meta-llama/llama-4-scout-17b-16e-instruct"),
    tools=[
        YFinanceTools(
            stock_price=True,
            analyst_recommendations=True,
            stock_fundamentals=True,
        )
    ],
    instructions=[
        "Retrieve key financial metrics and explain their impact on stock performance.",
        "Use tables for clarity.",
    ],
)

# Aggregator Agent
aggregator_agent = Agent(
    model=Groq(id="meta-llama/llama-4-scout-17b-16e-instruct"),
    agents=[news_agent, finance_agent],
    instructions=[
        "Combine the financial news and metrics into a summary.",
        "Provide investment recommendations.",
    ],
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
