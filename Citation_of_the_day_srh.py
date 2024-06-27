from typing import List
import os
import getpass
from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langserve import add_routes
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
system_template = ("Give a citation/quote based on people plan today: {What are you doing today?}:"
                   "For example, if people go working today, you can write: Successful people are not gifted; they just work hard, then succeed on purpose. — G.K. Nielson")


prompt_template = ChatPromptTemplate.from_messages([
    ('system', system_template),
    ('system', 'based on people plan today: {What are you doing today?}')
])

# 2. Create model
model = ChatOpenAI()

# Create parser to get only the response
parser = StrOutputParser()

# Chain our steps
chain = prompt_template | model | parser


# App definition
app = FastAPI(
  title="LangChain Server",
  version="1.0",
  description="A simple API server using LangChain's Runnable interfaces",
)

# Adding chain route

add_routes(
    app,
    chain,
    path="/chain",
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8555)

# Run the app on: http://localhost:8555/chain/playground/