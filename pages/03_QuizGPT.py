import json
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
import streamlit as st
from langchain.retrievers import WikipediaRetriever
from langchain.schema import BaseOutputParser, output_parser

class JsonOutputParser(BaseOutputParser):
    def parse(self, text):
        text = text.replace("```", "").replace("json", "")
        return json.loads(text)


function = {
    "name": "create_quiz",
    "description": "Function that takes a list of questions and answers and returns a quiz in JSON format.",
    "parameters": {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                        },
                        "answers": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "answer": {
                                        "type": "string",
                                    },
                                    "correct": {
                                        "type": "boolean",
                                    },
                                },
                                "required": ["answer", "correct"],
                            },
                        },
                    },
                    "required": ["question", "answers"],
                },
            }
        },
        "required": ["questions"],
    },
}

output_parser = JsonOutputParser()

st.set_page_config(
    page_title="QuizGPT",
    page_icon="❓",
)


st.title("QuizGPT")


llm = ChatOpenAI(
    temperature=0.1,
    model="gpt-3.5-turbo-1106",
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
).bind(
    function_call={
        "name": "create_quiz",
    },
    functions=[function],
)


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


def get_questions_prompt():
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
    You are a helpful assistant that is role playing as a teacher.
    
    Based ONLY on the following context, make 5 (FIVE) questions to test the user's knowledge about the text.
    
    Each question should have 4 answers, three of them must be incorrect and one should be correct.
    
    Adjust the difficulty level of the questions based on the given difficulty: {difficulty}.
    
    Context: {context}
    """,
            )
        ]
    )


@st.cache_data(show_spinner="Loading file...")
def split_file(file):
    file_content = file.read()
    file_path = f"./.cache/quiz_files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    return docs


@st.cache_data(show_spinner="Making quiz...")
def run_quiz_chain(_docs, topic, difficulty):
    questions_prompt = get_questions_prompt()

    chain = questions_prompt | llm

    result = chain.invoke({"context": format_docs(_docs), "difficulty": difficulty})

    formatted_quiz = result.additional_kwargs["function_call"]["arguments"]

    formatted_quiz_obj = json.loads(formatted_quiz)

    return formatted_quiz_obj
    

@st.cache_data(show_spinner="Searching Wikipedia...")
def wiki_search(term):
    retriever = WikipediaRetriever(top_k_results=2)
    docs = retriever.get_relevant_documents(term)
    return docs


with st.sidebar:
    docs = None
    topic = None
    choice = st.selectbox(
        "Choose what you want to use.",
        (
            "File",
            "Wikipedia Article",
        ),
    )
    if choice == "File":
        file = st.file_uploader(
            "Upload a .docx , .txt or .pdf file",
            type=["pdf", "txt", "docx"],
        )
        if file:
            docs = split_file(file)
    else:
        topic = st.text_input("Search Wikipedia...")
        if topic:
            docs = wiki_search(topic)


if not docs:
    st.markdown(
        """
    Welcome to QuizGPT.
                
    I will make a quiz from Wikipedia articles or files you upload to test your knowledge and help you study.
                
    Get started by uploading a file or searching on Wikipedia in the sidebar.
    """
    )
else:
    option = st.selectbox(
        "Choose the difficulty level",
        ("Hard", "Normal", "Easy"),
    )

    response = run_quiz_chain(docs, topic if topic else file.name, option)

    with st.form("questions_form"):
        for idx, question in enumerate(response["questions"][:5]):
            st.write(question["question"])
            value = st.radio(
                "Select an option.",
                [answer["answer"] for answer in question["answers"]],
                index=None,
                key=f"{question['question']}_{idx}"  
            )
            congratulation = True
            if {"answer": value, "correct": True} in question["answers"]:
                st.success("Correct!")
            elif value is not None:
                st.error("Wrong!")
                congratulation = False
        
        submit_button = st.form_submit_button()

    if submit_button:
        if congratulation == True:
            st.balloons()
        else:
            if st.button("Try Again"):
                st.experimental_rerun()