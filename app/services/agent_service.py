import os
import re
from dotenv import load_dotenv
from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.yfinance import YFinanceTools

load_dotenv()
groq_api_key = os.environ["GROQ_API_KEY"]
print(f"DEBUG: GROQ_API_KEY loaded: {groq_api_key[:5]}...{groq_api_key[-5:]}")

# Utility to clean markdown
def clean_markdown(text: str) -> str:
    """Cleans and normalizes markdown output for better formatting."""
    text = re.sub(r'\r\n|\r', '\n', text)  
    text = re.sub(r'\n{3,}', '\n\n', text)  
    return text.strip()


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

# Finance Agent
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
        "Retrieve key financial metrics, including Revenue, Net Income, Gross Margin, Operating Expenses, Earnings Per Share (EPS), P/E ratio, Market Cap, Dividend Yield, Beta, and Alpha.",
        "For relevant metrics (e.g., Revenue, Net Income, EPS), also retrieve and provide their Year-over-Year (YoY) change percentages.",
        "Use tables for clarity.",
        "Focus on how these metrics impact stock performance."
    ]
)

# Aggregator Agent
aggregator_agent = Agent(
    model=Groq(id="llama-3.3-70b-versatile"),
    agents=[news_agent, finance_agent],
    # instructions=[
    #     "You are a sophisticated AI financial analyst. Your task is to combine the inputs from the News Agent and Finance Agent into a comprehensive, well-structured stock report.",
    #     "The report MUST be presented entirely in Markdown format.",
    #     "**Strictly follow this numbered section structure and formatting:**",
    #     "",
    #     "# [Stock Name] Stock Analysis: Key Insights for Investors",
    #     "",
    #     "## 1. Introduction to [Stock Name]",
    #     "   - Provide a concise introduction to the company, its business, and its market position.",
    #     "",
    #     "## 2. Critical News & Market Factors",
    #     "   - Use clear bullet points for each critical news item or market factor identified.",
    #     "   - Explain how these factors are impacting or could impact the stock performance.",
    #     "   - Example factors: product launches, supply chain issues, competitive landscape.",
    #     "",
    #     "## 3. Key Financial Metrics (e.g., Fiscal Year End 2023 or Latest Reporting Period)",
    #     "   - **IMPORTANT: Present ALL financial metrics in a strict Markdown table format with a header and separator line.**",
    #     "   - The table MUST have exactly these three columns: `Metric`, `Value`, and `Change (YoY)`.",
    #     "   - If a 'Change (YoY)' value is not available for a specific metric, display 'N/A' or '-'.",
    #     "   - **Example of the REQUIRED table structure (including header and separator):**",
    #     "     ```",
    #     "     | Metric     |  Value    | Change (YoY) |",
    #     "     | :-----     |  :----    | :----------- |",
    #     "     | Revenue    |  $394.33B |    7.8%      |",
    #     "     | Net Income |  $99.8B   |    5.4%      |",
    #     "     | EPS | $5.67|  9.1%     |      -       |",
    #     "     | P/E Ratio  |  29.56    |     N/A      |",
    #     "     | Market Cap |  $2.97T   |      -       |",
    #     "     ```",
    #     "   - Include metrics such as: Revenue, Net Income, Gross Margin, Operating Expenses, Earnings Per Share (EPS), P/E ratio, Market Cap, Dividend Yield, Beta, and Alpha. Add others if available and relevant, ensuring all three columns are populated.",
    #     "",
    #     "## 4. Investment Recommendations",
    #     "   - Provide distinct sections for long-term and short-term investment outlooks.",
    #     "",
    #     "   ### A. Long-term Outlook:",
    #     "      - Use clear bullet points for each recommendation (Buy/Hold) and the detailed reasons supporting it.",
    #     "      - **Buy Recommendation:** Focus on aspects like strong brand loyalty, diversified product portfolio, continuous innovation, and growth potential (e.g., services segment).",
    #     "      - **Hold Recommendation:** Focus on factors like historical ability to navigate market fluctuations and commitment to shareholder value (dividends, buybacks).",
    #     "",
    #     "   ### B. Short-term Outlook:",
    #     "      - Use clear bullet points for each recommendation (Neutral/Sell) and the detailed reasons supporting it.",
    #     "      - **Neutral Recommendation:** Focus on volatility, product launch cycles, quarterly earnings, and global economic conditions.",
    #     "      - **Sell Recommendation:** Focus on realizing short-term profits or adjusting portfolio based on risk tolerance.",
    #     "",
    #     "## 5. Conclusion",
    #     "   - Provide a concise summary of the key takeaways for both long-term and short-term investors.",
    #     "   - **Crucially, include the following disclaimer at the very end of the conclusion:** 'Always conduct thorough personal research or consult with a qualified financial advisor before making any investment decisions.'",
    #     "",
    #     "**General Formatting Guidelines:**",
    #     "   - Use bolding (`**text**`) for emphasis where appropriate (e.g., recommendation types).",
    #     "   - Ensure consistent Markdown headings and lists throughout the report.",
    #     "   - Avoid any introductory or concluding sentences outside the specified Markdown structure.",
    #     "   - Do NOT include any code blocks or examples in the final output, unless specifically part of the requested Markdown table example.",
    #     "   - Make sure the report is well-organized and easy to read."
    # ]
instructions = [
    "You are a sophisticated AI financial analyst. Your task is to generate a professional Markdown stock analysis report based on inputs from News Agent and Finance Agent.",
    "Always use accurate financial terminology and clearly explain stock market actions, trends, and metrics. The output should serve both retail and institutional investors.",
    "",
    "**STRICTLY FOLLOW THIS STRUCTURE AND FORMAT (DO NOT DEVIATE):**",
    "",
    "# [Stock Name] Stock Analysis: Key Insights for Investors",
    "",
    "## 1. Introduction to [Stock Name]",
    "- Brief company overview: industry, business model, and key markets.",
    "- Include major recent developments (e.g., mergers, acquisitions, product lines, partnerships, or expansions).",
    "",
    "## 2. Critical News & Market Factors",
    "- Use bullet points.",
    "- Begin with macroeconomic indicators: interest rates, inflation, GDP trends, currency fluctuations.",
    "- Then cover micro-level news: earnings, layoffs, product launches, partnerships, buybacks, stock splits, dividends, regulations, and geopolitical risks.",
    "- End each bullet with a **bullish** or **bearish** tag in parentheses if applicable.",
    "",
    "## 3. Key Financial Metrics (Latest Annual or Quarterly)",
    "- Start with this note (modify the date as needed):",
    "  **Latest Financial Metrics (e.g., FY 2024 or Q4 FY 2024):**",
    "- Use the **Markdown table** format exactly as shown below:",
    "```",
    "| Metric              | Value     | Change (YoY) |",
    "| :----------------- | :-------- | :----------- |",
    "```",
    "- Include the following metrics if available:",
    "  - Revenue, Net Income, EPS, Gross Margin, Operating Margin, Operating Expenses",
    "  - Free Cash Flow, P/E Ratio, PEG Ratio, ROE, ROA, Market Cap, Dividend Yield, Beta, Alpha, Debt/Equity",
    "- For missing data, use `N/A` or `-`.",
    "- Write percentage changes as plain values (e.g., `2.1%` instead of `2.1% increase`).",
    "",
    "## 4. Investment Recommendations",
    "- Clearly distinguish between **long-term**, **mid-term**, and **short-term** outlooks.",
    "- Provide a recommendation and brief rationale for each.",
    "",
    "### A. Long-term Outlook:",
    "- Use one: **Buy**, **Hold**, or **Avoid**.",
    "- Justify using factors like innovation, market leadership, balance sheet strength, dividend record, and future growth potential.",
    "",
    "### B. Mid-term Outlook:",
    "- Use one: **Accumulate**, **Hold**, or **Risk Watch**.",
    "- Base this on upcoming product cycles, macroeconomic outlook, liquidity, or earnings momentum.",
    "",
    "### C. Short-term Outlook:",
    "- Use one: **Trade**, **Sell**, or **Neutral**.",
    "- Mention immediate technical factors, earnings reports, and analyst sentiment.",
    "- Recommend a **stop-loss strategy** (e.g., 5–7%) if relevant.",
    "",
    "## 5. Risk Factors",
    "- Use bullet points.",
    "- Include legal issues, lawsuits, economic headwinds, debt burdens, and competitive threats.",
    "- Mention the following explicitly when relevant:",
    "  * **Interest rate sensitivity**",
    "  * **Regulatory uncertainty**",
    "  * **Supply chain issues**",
    "  * **Currency volatility**",
    "",
    "## 6. Conclusion",
    "- Summarize recommendations clearly for each investor type (short-, mid-, and long-term).",
    "- Highlight final verdicts using **bold**:",
    "  * Example: **Buy**, **Hold**, **Avoid**.",
    "",
    "**DISCLAIMER: Always conduct your own research or consult a licensed financial advisor before making investment decisions.**",
    "",
    "## Output Rules",
    "- Use clean, readable Markdown formatting only.",
    "- Never add headings, comments, or text outside the defined structure.",
    "- Do not include code blocks except for tables.",
    "- Do not include raw JSON, HTML, or XML.",
    "- Use consistent formatting for lists, tables, and bold/italic text.",
]
)

# Final Query Function
def query_aggregator_agent(stock_name: str) -> str:
    """Fetches the full markdown stock analysis report from the aggregator agent."""
    response = aggregator_agent.run(message=f"Analyze {stock_name} stock.")

    if isinstance(response, dict) and "content" in response:
        content = response["content"]
    elif hasattr(response, "content"):
        content = response.content
    elif isinstance(response, str):
        content = response
    else:
        raise ValueError("Failed to get response content.")

    return clean_markdown(content)
