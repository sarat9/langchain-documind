from langchain.agents import create_csv_agent


from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_types import AgentType

from dotenv import load_dotenv, find_dotenv
import os

# Load environment variables from .env file
load_dotenv(find_dotenv())


agent = create_csv_agent(
    OpenAI(temperature=0),
    "country_full.csv",
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
)


def chat_with_csv(query):
    print("QUERY: " + query)
    result = agent.run(query)
    return result


def main():
    while True:
        input_data = input("Enter input (or 'exit' to quit): ")
        
        if input_data.lower() == 'exit':
            print("Exiting the program...")
            break
        
        print("Query: " + input_data)
        result = chat_with_csv(input_data)
        print("Result:", result)
        print()
    
    print("Program finished.")

if __name__ == "__main__":
    main()
