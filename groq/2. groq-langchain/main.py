import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv(os.path.join(os.path.dirname(__file__), "../../", ".env"))

# Initialize Groq Chat
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name=os.getenv("GROQ_MODEL_NAME"),
)

# Invoke the model
response = llm.invoke("What is the capital of France?")

# Print the content of the response
print(response.content)
