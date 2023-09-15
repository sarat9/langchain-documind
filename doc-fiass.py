from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from utils.pdf_text_extractor import extract_text_from_pdf
load_dotenv()

knowledge_base = None

def load_pdf_as_knowledge_base(path_to_pdf_file):
    global knowledge_base
    
    pdf_text = extract_text_from_pdf(path_to_pdf_file)
    # split into chunks
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(pdf_text)

    
    # create embeddings
    embeddings = OpenAIEmbeddings()
    knowledge_base = FAISS.from_texts(chunks, embeddings)


def chat_with_pdf(query):
    global knowledge_base
    # show user input
    response = None
    user_question = query
    if user_question:
        docs = knowledge_base.similarity_search(user_question)
        
        llm = OpenAI()
        chain = load_qa_chain(llm, chain_type="stuff")
        with get_openai_callback() as cb:
            response = chain.run(input_documents=docs, question=user_question)
            # print("Charges for the API:::")
            # print(cb)
        
    return response

def process_input(input_data):
    result = chat_with_pdf(input_data)
    return result

def main():
    print("Loading a Document...")
    load_pdf_as_knowledge_base('TCS_Concall.pdf')
    print("Start chatting with the Document...")

    while True:
        input_data = input("Enter input (or 'exit' to quit): ")
        
        if input_data.lower() == 'exit':
            print("Exiting the program...")
            break
        
        print("Query: " + input_data)
        result = process_input(input_data)
        print("Result:", result)
        print()
        print()
        print()
        print()
    
    print("Program finished.")

if __name__ == "__main__":
    main()
