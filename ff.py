from langchain.utilities import DuckDuckGoSearchAPIWrapper, WikipediaAPIWrapper
from langchain.agents.agent_toolkits import FileManagementToolkit
import yfinance
import json
import streamlit as st
import openai as client
import re

api_pattern = r'sk-.*'

assistant_pattern = r'asst-.*'

st.set_page_config(
    page_title="Assistants",
    page_icon="🖥️",
)

def get_search_wikipedia(inputs):
  wiki = WikipediaAPIWrapper()
  query = inputs["query"]
  return wiki.run(query)
  
def get_search_duck(inputs):
  ddg = DuckDuckGoSearchAPIWrapper()
  query = inputs["query"]
  return ddg.run(query)

# def get_txt(inputs):
#   tools = FileManagementToolkit(
#       root_dir=str(".txt/"),
#       selected_tools=["read_file", "write_file", "list_directory"],
#     ).get_tools()
#   read_tool, write_tool, list_tool = tools
#   output = inputs["output"]
#   return write_tool.invoke({"file_path": "output.txt", "text": output})

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

functions = [
  {
    "type": "function",
    "function": {
      "name": "get_search_wikipedia",
      "description": "Use this tool to find the website for the given query",
      "parameters": {
        "type": "object",
        "properties": {
          "query": {
            "type": "string",
            "description": "The query to search for",
          },
        },
        "required": ["query"],
      },
    },
  },
  {
    "type": "function",
    "function": {
      "name": "get_search_duck",
      "description": "Use this tool to find the website for the given query",
      "parameters": {
        "type": "object",
        "properties": {
          "query": {
            "type": "string",
            "description": "The query to search for",
          },
        },
        "required": ["query"],
      },
    },
  },
  # {
  #   "type": "function",
  #   "function": {
  #     "name": "get_txt",
  #     "description": "Converts the output to a text file",
  #     "parameters": {
  #       "type": "object",
  #       "properties": {
  #         "output": {
  #           "type": "string",
  #           "description": "The output to convert to a text file",
  #         },
  #       },
  #       "required": ["output"],
  #     },
  #   },
  # },
]

if "api_key" not in st.session_state:
  st.session_state["api_key"] = None

if "api_key_bool" not in st.session_state:
  st.session_state["api_key_bool"] = False

if "assistant_id" not in st.session_state:
  st.session_state["assistant_id"] = None
  
def save_assistant_id(assistant_id):
  st.session_state["assistant_id"] = assistant_id

def save_api_key(api_key):
    st.session_state["api_key"] = api_key
    
def make_assistant(api_key):
  assistant = client.beta.assistants.create(
        name="Search Assistant",
        instructions="You search for user questions and convert the results into a txt file",
        model="gpt-4-1106-preview",
        tools=functions,
      )
  
with st.sidebar:
  api_key = st.text_input("Enter your OpenAI API key")
  
  if api_key:
    if api_key != st.session_state["api_key"]:
      make_assistant(api_key)
    save_api_key(api_key)
    if not re.match(api_pattern, api_key):
        st.write("API_KEY가 올바르지 않습니다.")
    else:
        st.write("API_KEY가 올바르게 저장되었습니다.")
  
  api_key_button = st.button("api키 저장")
  
  if api_key_button:
    save_api_key(api_key)
    st.session_state["store_click"] = True
    if api_key == "":
        st.write("API_KEY를 넣어주세요.")
            
  assistant_id = st.text_input("Enter your assistant ID")
  
  if assistant_id:
    save_assistant_id(assistant_id)
    if not re.match(assistant_pattern, assistant_id):
      st.write("Assistant_ID가 올바르지 않습니다.")
    else:
      st.write("Assistant_ID가 올바르게 저장되었습니다.")
      
  assistant_id_button = st.button("어시스턴트 저장")
  
  if assistant_id_button:
    save_assistant_id(assistant_id)
    if assistant_id == "":
        st.write("Assistant_ID를 넣어주세요.")
    
st.markdown(
    """
    # Assistants
"""
)

def get_run(run_id, thread_id):
    return client.beta.threads.runs.retrieve(
        run_id=run_id,
        thread_id=thread_id,
    )

def get_messages(thread_id):
    messages = client.beta.threads.messages.list(thread_id=thread_id)
    messages = list(messages)
    messages.reverse()
    for message in messages:
        print(f"{message.role}: {message.content[0].text.value}")

def make_thread(message):
  thread = client.beta.threads.create(
      messages=[
        {
          "role": "user",
          "content": message,
        }
      ]
    )
  return thread

def run_thread(thread_id, assistant_id):
  run = client.beta.threads.runs.create(
      thread_id=thread_id,
      assistant_id=assistant_id,
    )
  return run

if assistant_id:
  send_message("I'm ready! Ask away!", "assistant", save=False)
  paint_history()
  message = st.text_input("Ask a question")
  if message:
    thread = make_thread(message)
    
    run_thread(thread.id, assistant_id)
    get_messages(thread.id)
else:
  st.session_state["messages"] = []