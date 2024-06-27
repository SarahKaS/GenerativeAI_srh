from typing import List
import os
import getpass
from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langserve import add_routes

# Passwords definition
# pass_langChain =
# pass_openAI =

# Set environment variables
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ["LANGCHAIN_API_KEY"] = pass_langChain # getpass.getpass()


# Verify the environment variables are set
print("LANGCHAIN_TRACING_V2:", os.environ['LANGCHAIN_TRACING_V2'])
print("LANGCHAIN_API_KEY:", os.environ['LANGCHAIN_API_KEY'])

os.environ["OPENAI_API_KEY"] = pass_openAI  # getpass.getpass()


# Create prompt template
system_template = ("You are a life coach. You encourage healthy, proactive and happy life. This include eating healthy, sleeping well and make exercise."
                   "You response will be short (maximum 60 words)"
                   "Your goal is to give advices and encouragements to motivate people based on his feelings today: {How are you feeling today?}:")

prompt_template = ChatPromptTemplate.from_messages([
    ('system', system_template),
    ('system', 'based on his feelings today: {How are you feeling today?}')
])


# 2. Choose model
model = ChatOpenAI()


# Create parser to get only the response
parser = StrOutputParser()


# Chain our steps
chain = prompt_template | model | parser


# App definition
app = FastAPI(
  title = "LangChain Server",
  version = "1.0",
  description = "A simple API server using LangChain's Runnable interfaces",
)


# Adding chain route
add_routes(
    app,
    chain,
    path="/chain",
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8005)

# Run the app on: http://localhost:8005/chain/playground/