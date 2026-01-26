import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import PIIMiddleware
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

load_dotenv(os.path.join(os.path.dirname(__file__), "../../", ".env"))

# Initialize the model
model = ChatGroq(
    model_name=os.getenv("GROQ_MODEL_NAME"),
    temperature=0,
    groq_api_key=os.getenv("GROQ_API_KEY"),
)

# Mock tools
@tool
def customer_service_tool(query: str) -> str:
    """A tool for handling customer service inquiries."""
    return f"Processed customer inquiry: {query}"

@tool
def email_tool(email: str, subject: str, message: str) -> str:
    """A tool for sending emails."""
    return f"Email sent to {email} with subject: {subject}"

# Create the agent with PII Middleware
agent = create_agent(
    model=model,
    tools=[customer_service_tool, email_tool],
    middleware=[
        # Redact emails in user input before sending to model
        PIIMiddleware(
            "email",
            strategy="redact",
            apply_to_input=True,
        ),
        # Mask credit cards in user input
        PIIMiddleware(
            "credit_card",
            strategy="mask",
            apply_to_input=True,
        ),
        # Block API keys - raise error if detected
        PIIMiddleware(
            "api_key",
            detector=r"sk-[a-zA-Z0-9]{32}",
            strategy="block",
            apply_to_input=True,
        ),
    ],
)

# Test with sensitive information
content = "My email is john.doe@example.com and card is 5105-1051-0510-5100"
print(f"Original Input: {content}\n")

try:
    result = agent.invoke({
        "messages": [HumanMessage(content=content)]
    })
    
    # We can inspect the messages in the state to see how they were modified
    # The middleware modifies the input message before the model sees it
    processed_input = result["messages"][0].content
    print(f"Input as seen by Model (via Middleware): {processed_input}\n")
    
    final_answer = result["messages"][-1].content
    print(f"Agent Final Answer: {final_answer}")

except Exception as e:
    print(f"Agent blocked input: {e}")
