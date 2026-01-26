import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langchain_core.tools import tool

load_dotenv(os.path.join(os.path.dirname(__file__), "../../", ".env"))

model = ChatGroq(
    model_name=os.getenv("GROQ_MODEL_NAME"),
    temperature=0.1,
    groq_api_key=os.getenv("GROQ_API_KEY"),
)

@tool
def get_word_length(word: str) -> int:
    """Returns the length of a word."""
    return len(word)

tools = [get_word_length]

agent = create_agent(
    model, 
    tools=tools, 
    system_prompt="You are a helpful assistant."
)

inputs = {"messages": [{"role": "user", "content": "How many letters are in the word 'Pranesh'?"}]}
response = agent.invoke(inputs)

final_message = response["messages"][-1]
print(f"\nFinal Answer: {final_message.content}")
