from langchain import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma, Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings

from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain

from utils.pdf_text_extractor import extract_text_from_pdf

from dotenv import load_dotenv, find_dotenv
import os

# Load environment variables from .env file
load_dotenv(find_dotenv())
PINE_CONE_API_KEY = os.getenv("PINE_CONE_API_KEY")
PINE_CONE_ENVIRONMENT = os.getenv("PINE_CONE_ENVIRONMENT")

INDEX_NAME = "langchainstuff"


import pinecone
# Initialize Pinecone with your API key
pinecone.init(api_key=PINE_CONE_API_KEY, environment=PINE_CONE_ENVIRONMENT)

# load pdf file data
def load_data_to_pine_cone(path_to_pdf_file='path/to/your/file.pdf'):
    # Open the PDF file in read binary mode
    pdf_text = extract_text_from_pdf(path_to_pdf_file)

    text_splitter = CharacterTextSplitter(
        # separators=["\n\n", "\n", " ", ""], #split on them in order until the chunks are small enough.The default list is ["\n\n", "\n", " ", ""]
        chunk_size=900, # size of characters for each chunk
        chunk_overlap=80, # buffer overlap chunk of previous characters
        length_function=len
    )
    # chunks = text_splitter.split_documents([pdf_text])
    chunks = text_splitter.create_documents([pdf_text])
    print(chunks)

    # create embeddings
    embeddings = OpenAIEmbeddings()

    # Loading to pine cone
    DIMENSIONS = 1536

    print("Loading to PineCone...")
    # https://docs.pinecone.io/reference/create_index
    # Create and configure index if doesn't already exist
        
    if INDEX_NAME not in pinecone.list_indexes():
        print("Index not found.. creating index..")
        pinecone.create_index(
            name=INDEX_NAME, 
            metric="cosine",
            dimension=DIMENSIONS)
        docsearch = Pinecone.from_documents(chunks, embeddings, index_name=INDEX_NAME)
    else:
        print("Index found.. using existing index..")
        docsearch = Pinecone.from_existing_index(INDEX_NAME, embeddings)
        print("Loading Done...")


def chat_with_pdf(query):
    print(query)
    embeddings = OpenAIEmbeddings()
    docsearch = Pinecone.from_existing_index(INDEX_NAME, embeddings)
    llm = OpenAI(temperature=0)
    chain = load_qa_chain(llm, chain_type="stuff")
    docs = docsearch.similarity_search(query)
    result = chain.run(input_documents=docs, question=query)
    return result



def process_input(input_data):
    result = chat_with_pdf(input_data)
    return result

def main():
    load_data_to_pine_cone('TCS_Concall.pdf')
    while True:
        input_data = input("Enter input (or 'exit' to quit): ")
        
        if input_data.lower() == 'exit':
            print("Exiting the program...")
            break
        
        print("Query: " + input_data)
        result = process_input(input_data)
        print("Result:", result)
        print()
    
    print("Program finished.")

if __name__ == "__main__":
    main()
