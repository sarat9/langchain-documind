from langchain_openai import OpenAI
from langchain_community.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
import os

from dotenv import load_dotenv, find_dotenv
# Load environment variables from .env file
load_dotenv(find_dotenv())
API_KEY = os.getenv('OPENAI_API_KEY')

# Setup database
db = SQLDatabase.from_uri(
    f"postgresql+psycopg2://postgres:{os.getenv('DB_PASSWORD')}@localhost:5432/{os.getenv('DB_NAME')}",
)

# setup llm
llm = OpenAI(temperature=0, openai_api_key=API_KEY)

# Create db chain
QUERY = """
Given an input question, first create a syntactically correct postgresql query to run, then look at the results of the query and return the answer.
Use the following format:

Question: Question here
SQLQuery: SQL Query to run
SQLResult: Result of the SQLQuery
Answer: Final answer here

{question}
"""

# Setup the database chain
db_chain = SQLDatabaseChain(llm=llm, database=db, verbose=True)


def get_prompt():
    print("Type 'exit' to quit")

    while True:
        prompt = input("Enter a prompt: ")

        if prompt.lower() == 'exit':
            print('Exiting...')
            break
        else:
            try:
                question = QUERY.format(question=prompt)
                print(db_chain.run(question))
            except Exception as e:
                print(e)

get_prompt()

