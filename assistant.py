from langchain.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper
from langchain.utilities.wikipedia import WikipediaAPIWrapper
from langchain.agents.agent_toolkits import FileManagementToolkit
import json
import streamlit as st
import openai as client
import re
import time

api_pattern = r'sk-.*'

assistant_pattern = r'asst_.*'

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

def get_txt(inputs):
  data = inputs["data"]
  with open("output.txt", "w", encoding="utf-8") as f:
    f.write(data)
  return "Data saved to output.txt"

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
            "description": "The query you will search for.Example query: Research about the XZ backdoor",
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
            "description": "The query you will search for.Example query: Research about the XZ backdoor",
          },
        },
        "required": ["query"],
      },
    },
  },
  {
    "type": "function",
    "function": {
      "name": "get_txt",
      "description": "Saves the given data to a file",
      "parameters": {
        "type": "object",
        "properties": {
          "data": {
            "type": "string",
            "description": "The data to save",
          },
        },
        "required": ["data"],
      },
    },
  },
]

if "api_key" not in st.session_state:
  st.session_state["api_key"] = None

if "api_key_bool" not in st.session_state:
  st.session_state["api_key_bool"] = False

if "assistant_id" not in st.session_state:
  st.session_state["assistant_id"] = None
  
if "assistant_id_bool" not in st.session_state:
  st.session_state["assistant_id_bool"] = False
  
def save_assistant_id(assistant_id):
  st.session_state["assistant_id"] = assistant_id

def save_api_key(api_key):
    st.session_state["api_key"] = api_key
    
def make_assistant():
  assistant = client.beta.assistants.create(
        name="Search Assistant",
        instructions="Please provide a query to search on Wikipedia or DuckDuckGo. For example, say 'Search Wikipedia for XZ backdoor'",
        model="gpt-4-1106-preview",
        tools=functions,
      )
  
  
with st.sidebar:
  api_key = st.text_input("Enter your OpenAI API key")
  
  if api_key:
    if api_key != st.session_state["api_key"]:
      make_assistant()
    else:
      st.write("동일한 API_KEY입니다")
      
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
      st.session_state["assistant_id_bool"] = True
      
  assistant_id_button = st.button("어시스턴트 저장")
  
  if assistant_id_button:
    save_assistant_id(assistant_id)
    if assistant_id == "":
        st.write("Assistant_ID를 넣어주세요.")
  
  st.write("https://github.com/su2minig/gpt-streamlit/tree/main")
    
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

functions_map = {
  "get_search_wikipedia": get_search_wikipedia,
  "get_search_duck": get_search_duck,
  "get_txt": get_txt,
}

def get_tool_outputs(run_id, thread_id):
    run = get_run(run_id, thread.id)
    outputs = []
    st.write(run)
    for action in run.required_action.submit_tool_outputs.tool_calls:
      action_id = action.id
      function = action.function
      print(f"Calling function: {function.name} with arg {function.arguments}")
      outputs.append(
          {
              "output": functions_map[function.name](json.loads(function.arguments)),
              "tool_call_id": action_id,
          }
      )
    return outputs
  
def submit_tool_outputs(run_id, thread_id):
  outpus = get_tool_outputs(run_id, thread_id)
  return client.beta.threads.runs.submit_tool_outputs(
      run_id=run_id, thread_id=thread_id, tool_outputs=outpus
  )

if assistant_id and (st.session_state["assistant_id_bool"]==True):
  send_message("I'm ready! Ask away!", "assistant", save=False)
  paint_history()
  message = st.text_input("Ask a question")
  if message:
    send_message(message, "human")
    thread = make_thread(message)
    
    run = run_thread(thread.id, assistant_id)
    st.write(run.id)
    if run.id != "":
      run = get_run(run.id, thread.id)
      if run.status == "completed":
        st.success("completed")
        get_messages(thread.id)
        with open("output.txt", "rb") as f:
          btn = st.download_button(
            label="Download",
            data=f,
            file_name="output.txt",
            mime="text/plain",
          )
    elif run.status == "in_progress":
      with st.status("running"):
        st.write(run.status)
        time.sleep(5)
        st.rerun()
    elif run.status == "requires_action":
      with st.status("requires_action"):
        submit_tool_outputs(run.id, thread.id)
        time.sleep(5)
        st.rerun()
else:
  st.session_state["messages"] = []