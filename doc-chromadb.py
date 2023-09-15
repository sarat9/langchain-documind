from langchain import OpenAI,ConversationChain,LLMChain, PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferWindowMemory, ConversationBufferMemory, ChatMessageHistory
from langchain.vectorstores import Chroma, Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings

from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
# Import chroma as the vector store 
from langchain.vectorstores import Chroma

# Import vector store stuff
from langchain.agents.agent_toolkits import (
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo
)

from dotenv import load_dotenv, find_dotenv

# Load environment variables from .env file
load_dotenv(find_dotenv())

def ask_question_with_chromadb(query):

    llm = OpenAI(temperature=0)

    data = "The future quater are forcated to grow at 24% upside in retail. The growth in retail is 24%. The healthcare is expecte dto grow at 12%. We are anticipating good numbers in future. The blockchain project is in progress and has a potential of 2400 million dollar deal"
    text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", " ", ""], #split on them in order until the chunks are small enough.The default list is ["\n\n", "\n", " ", ""]
            chunk_size=1000, # size of characters for each chunk
            chunk_overlap=0, # buffer overlap chunk of previous characters
            length_function=len
        )
    chunks = text_splitter.create_documents([data])
    # create embeddings
    embeddings = OpenAIEmbeddings()

    store = Chroma.from_documents(chunks, embeddings, collection_name='annualreport')

    # Create vectorstore info object - metadata repo?
    vectorstore_info = VectorStoreInfo(
        name="annual_report",
        description="a banking annual report as a pdf",
        vectorstore=store
    )

    # Convert the document store into a langchain toolkit
    toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info)

    # Add the toolkit to an end-to-end LC
    agent_executor = create_vectorstore_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=True
    )
    print("Query : ")
    print(query)
    result = None
    # If the user hits enter
    if query:
        # Then pass the prompt to the LLM
        response = agent_executor.run(query)
        print("Answer : ")
        print(response)
        result = response
        print("Search : ")
        # Find the relevant pages
        search = store.similarity_search_with_score(prompt) 
        # Write out the first 
        print(search[0][0].page_content) 

    return result



ask_question_with_chromadb("is there any prediction for retail?")