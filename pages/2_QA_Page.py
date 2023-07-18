import langchain
import openai
import streamlit_chat
from langchain import OpenAI
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings, CohereEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import streamlit as st
from streamlit_chat import message
from langchain import Cohere
import os
from langchain.prompts import PromptTemplate


def append_to_file(file_path, text):
    try:
        with open(file_path, 'w', encoding="utf-8") as file:
            file.write(str(text))
        print("Parameters appended to the file successfully.")
    except IOError:
        print("An error occurred while appending parameters to the file.")


def read_file(file_path):
    with open(file_path, 'r', encoding="utf-8") as file:
        response = file.read()
        return response


def splitter(response):
    llm = (response.split('^')[0])
    api = (response.split('^')[1])
    file = (response.split('^')[2])
    return llm, api, file


def get_embeddings(documents, llm, api):
    if documents != '':
        if llm == 'OpenAI':
            os.environ["OPENAI_API_KEY"] = api
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
            documents = text_splitter.split_documents(documents)
            embeddings = OpenAIEmbeddings()
            vectorstore = Chroma.from_documents(documents, embeddings)
            memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
            qa2 = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0.6), vectorstore.as_retriever(),
                                                        memory=memory)
            return qa2
        else:
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
            documents = text_splitter.split_documents(documents)
            embeddings = CohereEmbeddings(cohere_api_key=api)
            vectorstore = Chroma.from_documents(documents, embeddings)
            memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
            qa2 = ConversationalRetrievalChain.from_llm(Cohere(cohere_api_key=api), vectorstore.as_retriever(),
                                                        memory=memory)
            return qa2


def file_loader(filename):
    if '.txt' in filename:
        loader = TextLoader(r"filename", encoding="utf-8")
        documents = loader.load()
        return documents
    else:
        return ''


def get_text():
    if llm == 'Cohere':
        input_text = st.text_input("Query: ", "", key="input")
        return input_text
    else:
        input_text = st.text_input("Query: ", "Hey!", key="input")
        return input_text


st.set_page_config(page_title="Chatbot - Private Docs", page_icon=":robot:")
st.header("Chatbot - Private Docs")
llm, api, text = splitter(read_file('pages/file.txt'))
append_to_file('pages/tempfile.txt', text)
docs = TextLoader(r"pages/tempfile.txt", encoding="utf-8").load()
qa = get_embeddings(docs, llm, api)
conversation_str = ""

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []

query = get_text()
result = qa({"question": query})

if query:
    conversation_str = conversation_str + '\nHuman: ' + result['question']
    conversation_str = conversation_str + '\nBot: ' + result['answer']
    st.session_state.past.append(result['question'])
    st.session_state.generated.append(result['answer'])

if st.session_state["generated"]:
    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")

# del st.session_state["api_key"]
