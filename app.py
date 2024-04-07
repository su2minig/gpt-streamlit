import json

from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
import streamlit as st
from langchain.retrievers import WikipediaRetriever

st.set_page_config(
    page_title="QuizGPT",
    page_icon="❓",
)

st.title("QuizGPT")

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


@st.cache_data(show_spinner="Searching Wikipedia...")
def wiki_search(term):
    retriever = WikipediaRetriever(top_k_results=5)
    docs = retriever.get_relevant_documents(term)
    return docs

function = {
    "name": "create_quiz",
    "description": "function that takes a list of questions and answers and returns a quiz",
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

with st.sidebar:
    api_key = st.text_input("Enter your OpenAI API key")
    code = """
    import json
    from langchain.document_loaders import UnstructuredFileLoader
    from langchain.text_splitter import CharacterTextSplitter
    from langchain.chat_models import ChatOpenAI
    from langchain.prompts import ChatPromptTemplate
    from langchain.callbacks import StreamingStdOutCallbackHandler
    import streamlit as st
    from langchain.retrievers import WikipediaRetriever
    from langchain.schema import BaseOutputParser, output_parser

    st.set_page_config(
        page_title="QuizGPT",
        page_icon="❓",
    )

    st.title("QuizGPT")

    def format_docs(docs):
        return "\n\n".join(document.page_content for document in docs)

    questions_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """"""
        You are a helpful assistant that is role playing as a teacher.
            
        Based ONLY on the following context make 10 (TEN) questions minimum to test the user's knowledge about the text with {level}.
        
        If the {level} is hard, make the quiz difficult to solve. If the {level} is easy, make the quiz easy to solve.
        
        Each question should have 4 answers, three of them must be incorrect and one should be correct.
            
        Use (o) to signal the correct answer.
            
        Question examples:
            
        Question: What is the color of the ocean?
        Answers: Red|Yellow|Green|Blue
            
        Question: What is the capital or Georgia?
        Answers: Baku|Tbilisi|Manila|Beirut
            
        Question: When was Avatar released?
        Answers: 2007|2001|2009|1998
            
        Question: Who was Julius Caesar?
        Answers: A Roman Emperor|Painter|Actor|Model
            
        Your turn!
            
        Context: {context}
        """""",
            )
        ]
    )

    function = {
        "name": "create_quiz",
        "description": "function that takes a list of questions and answers and returns a quiz",
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

    llm = ChatOpenAI(
        temperature=0.1,
        model="gpt-3.5-turbo-1106",
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()],
    ).bind(
        function_call={
            "name": "create_quiz",
        },
        functions=[
            function,
        ],
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
    def run_quiz_chain(level, _docs):
        chain = questions_prompt | llm
        
        response = chain.invoke({"level": level,"context": docs})
        response = json.loads(response.additional_kwargs["function_call"]["arguments"])
        return response


    @st.cache_data(show_spinner="Searching Wikipedia...")
    def wiki_search(term):
        retriever = WikipediaRetriever(top_k_results=5)
        docs = retriever.get_relevant_documents(term)
        return docs
    
    with st.sidebar:
        api_key = st.text_input("Enter your OpenAI API key")
        if api_key:
            llm = ChatOpenAI(
                api_key=api_key,
                temperature=0.1,
                model="gpt-3.5-turbo-1106",
                streaming=True,
                callbacks=[StreamingStdOutCallbackHandler()],
            ).bind(
                function_call={
                    "name": "create_quiz",
                },
                functions=[
                    function,
                ],
            )
        
        docs = None
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
            """"""
        Welcome to QuizGPT.
                    
        I will make a quiz from Wikipedia articles or files you upload to test your knowledge and help you study.
                    
        Get started by uploading a file or searching on Wikipedia in the sidebar.
        """"""
        )
    else:
        with st.form("Level"):
            level = st.radio(
                "Select the level of difficulty.",
                ["Easy","Hard"],
                index=None,
            )
            level_button = st.form_submit_button()
            if level_button:
                st.session_state["level"]=level
            
        st.write(level)
        if st.session_state["level"]!=None:
            response = run_quiz_chain(level, docs)
            with st.form("questions_form"):
                correct_answers = 0
                answered_questions = 0
                total_questions = len(response["questions"])
                answers = {}
                
                for question in response["questions"]:
                    st.write(question["question"])
                    value = st.radio(
                        "Select an option.",
                        [answer["answer"] for answer in question["answers"]],
                        index=None,
                        key=question["question"],
                    )
                    if value:
                        answered_questions += 1
                        answers[question["question"]] = value
                    
                    if {"answer": value, "correct": True} in question["answers"]:
                        correct_answers += 1
                        st.success("Correct!")
                    elif value is not None:
                        st.error("Wrong!")

                    if correct_answers == total_questions:
                        st.balloons()
                        st.success("Congratulations! You answered all questions correctly.")
                
                button = st.form_submit_button()
    """
    

    docs = None
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
            
    st.write("https://github.com/su2minig/gpt-streamlit")
    st.markdown("```python\n"+code+"\n```")

def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)

questions_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
    You are a helpful assistant that is role playing as a teacher.
         
    Based ONLY on the following context make 10 (TEN) questions minimum to test the user's knowledge about the text with {level}.
    
    If the {level} is hard, make the quiz difficult to solve. If the {level} is easy, make the quiz easy to solve.
    
    Each question should have 4 answers, three of them must be incorrect and one should be correct.
         
    Use (o) to signal the correct answer.
         
    Question examples:
         
    Question: What is the color of the ocean?
    Answers: Red|Yellow|Green|Blue
         
    Question: What is the capital or Georgia?
    Answers: Baku|Tbilisi|Manila|Beirut
         
    Question: When was Avatar released?
    Answers: 2007|2001|2009|1998
         
    Question: Who was Julius Caesar?
    Answers: A Roman Emperor|Painter|Actor|Model
         
    Your turn!
         
    Context: {context}
""",
        )
    ]
)


if api_key:
        llm = ChatOpenAI(
            api_key=api_key,
            temperature=0.1,
            model="gpt-3.5-turbo-1106",
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()],
        ).bind(
            function_call={
                "name": "create_quiz",
            },
            functions=[
                function,
            ],
        )


@st.cache_data(show_spinner="Making quiz...")
def run_quiz_chain(level, _docs):
    chain = questions_prompt | llm
    
    response = chain.invoke({"level": level,"context": docs})
    response = json.loads(response.additional_kwargs["function_call"]["arguments"])
    return response


if not docs:
    st.markdown(
        """
    Welcome to QuizGPT.
                
    I will make a quiz from Wikipedia articles or files you upload to test your knowledge and help you study.
                
    Get started by uploading a file or searching on Wikipedia in the sidebar.
    """
    )
else:
    with st.form("Level"):
        level = st.radio(
            "Select the level of difficulty.",
            ["Easy","Hard"],
            index=None,
        )
        level_button = st.form_submit_button()
        if level_button:
            st.session_state["level"]=level
        
    st.write(level)
    if st.session_state["level"]!=None:
        response = run_quiz_chain(level, docs)
        with st.form("questions_form"):
            correct_answers = 0
            answered_questions = 0
            total_questions = len(response["questions"])
            answers = {}
            
            for question in response["questions"]:
                st.write(question["question"])
                value = st.radio(
                    "Select an option.",
                    [answer["answer"] for answer in question["answers"]],
                    index=None,
                    key=question["question"],
                )
                if value:
                    answered_questions += 1
                    answers[question["question"]] = value
                
                if {"answer": value, "correct": True} in question["answers"]:
                    correct_answers += 1
                    st.success("Correct!")
                elif value is not None:
                    st.error("Wrong!")

                if correct_answers == total_questions:
                    st.balloons()
                    st.success("Congratulations! You answered all questions correctly.")
            
            button = st.form_submit_button()