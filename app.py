from langchain.document_loaders import SitemapLoader
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
import streamlit as st
from langchain.callbacks.base import BaseCallbackHandler

st.set_page_config(
    page_title="Assistants",
    page_icon="🖥️",
)

def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})
    
def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)
        
def paint_history():
    for message in st.session_state["messages"]:
        send_message(
            message["message"],
            message["role"],
            save=False,
        )
        
def load_memory(_):
    return memory.load_memory_variables({})["history"]

class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


with st.sidebar:
    url = st.text_input(
        "Write down a URL",
        placeholder="https://example.com",
    )
    api_key = st.text_input("Enter your OpenAI API key")
    st.write("https://github.com/su2minig/gpt-streamlit")
        
    code ="""
    
    """
    st.markdown("```python\n"+code+"\n```")
    
if not api_key:
    st.warning("Please provide an **:blue[OpenAI API Key]** on the sidebar.")

memory = ConversationBufferMemory(return_messages=True, memory_key="history")

answers_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Using ONLY the following context answer the user's question. If you can't just say you don't know, don't make anything up.
                                                        
            Then, give a score to the answer between 0 and 5.
            If the answer answers the user question the score should be high, else it should be low.
            Make sure to always include the answer's score even if it's 0.
            Context: {context}
                                                        
            Examples:
                                                        
            Question: How far away is the moon?
            Answer: The moon is 384,400 km away.
            Score: 5
                                                        
            Question: How far away is the sun?
            Answer: I don't know
            Score: 0    
            """,
        ),
        ("human", "{question}"),
    ]
)
if api_key:
    llm = ChatOpenAI(
        api_key=api_key,
        temperature=0.1,
        callbacks=[
            ChatCallbackHandler(),
            ],
    )


def get_answers(inputs):
    docs = inputs["docs"]
    question = inputs["question"]
    
    answers_chain = answers_prompt | llm
    return {
        "question": question,
        "answers": [
            {
                "answer": answers_chain.invoke(
                    {"question": question, "context": doc.page_content}
                ).content,
                "source": doc.metadata["source"],
                "date": doc.metadata["lastmod"],
            }
            for doc in docs
        ],
    }


choose_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Use ONLY the following pre-existing answers to answer the user's question.

            Use the answers that have the highest score (more helpful) and favor the most recent ones.

            Cite sources and return the sources of the answers as they are, do not change them.

            Answers: {answers}
            """,
        ),
        ("human", "{question}"),
    ]
)


def choose_answer(inputs):
    answers = inputs["answers"]
    question = inputs["question"]
    llm.streaming = True
    
    choose_chain = choose_prompt | llm
    condensed = "\n\n".join(
        f"{answer['answer']}\nSource:{answer['source']}\nDate:{answer['date']}\n"
        for answer in answers
    )
    return choose_chain.invoke(
        {
            "question": question,
            "answers": condensed,
        }
    )


def parse_page(soup):
    header = soup.find("header")
    footer = soup.find("footer")
    if header:
        header.decompose()
    if footer:
        footer.decompose()
    return (
        str(soup.get_text())
        .replace("\n", " ")
        .replace("\xa0", " ")
        .replace("CloseSearch Submit Blog", "")
    )


@st.cache_data(show_spinner="Loading website...")
def load_website(url):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=200,
    )
    loader = SitemapLoader(
        url,
        filter_urls=[
            "https://developers.cloudflare.com/ai-gateway/",
            "https://developers.cloudflare.com/vectorize/",
            "https://developers.cloudflare.com/workers-ai/",
        ],
        parsing_function=parse_page,
    )
    loader.requests_per_second = 2
    docs = loader.load_and_split(text_splitter=splitter)
    vector_store = FAISS.from_documents(docs, OpenAIEmbeddings(api_key=api_key))
    return vector_store.as_retriever()


st.markdown(
    """
    # SiteGPT
            
    Ask questions about the content of a website.
            
    Start by writing the URL of the website on the sidebar.
"""
)

    
def invoke_chain(question):
    result = chain.invoke(question)
    memory.save_context(
        {"input": question},
        {"output": result.content},
    )
    result = result.content.replace("$", "\$")
    return result

if url and api_key:
    if ".xml" not in url:
        with st.sidebar:
            st.error("Please write down a Sitemap URL.")
    else:
        retriever = load_website(url)
        send_message("I'm ready! Ask away!", "ai", save=False)
        paint_history()
        message = st.text_input("Ask a question to the website.")
        if message:
            send_message(message, "human")
            chain = (
                {
                    "docs": retriever,
                    "question": RunnablePassthrough(),
                }
                | RunnableLambda(get_answers)
                | RunnableLambda(choose_answer)
            )
            with st.chat_message("ai"):
                invoke_chain(message)
else:
    st.session_state["messages"] = []