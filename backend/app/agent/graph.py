import os
from typing import Annotated, TypedDict, List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from app.agent.tools import get_dataset_profile, clean_dataset, perform_eda, train_model

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    
tools = [get_dataset_profile, clean_dataset, perform_eda, train_model]
tool_node = ToolNode(tools)

llm_with_tools = None

def get_llm():
    global llm_with_tools
    if llm_with_tools is None:
        llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0)
        llm_with_tools = llm.bind_tools(tools)
    return llm_with_tools

def chatbot(state: AgentState):
    try:
        model = get_llm()
        response = model.invoke(state["messages"])
        return {"messages": [response]}
    except Exception as e:
        return {"messages": [BaseMessage(content=f"LLM Error: {str(e)}", type="ai")]}

graph_builder = StateGraph(AgentState)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", tool_node)

graph_builder.set_entry_point("chatbot")
graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge("tools", "chatbot")

agent = graph_builder.compile()

system_prompt = """You are DataMind, an AI data-science assistant. 
Your goal is to help users analyze datasets, clean data, and train ML models.
Use the provided tools to inspect datasets, perform EDA, clean them, and train models.
Always explain the findings from the tools clearly and ground your answers in the tool outputs.
If data is insufficient or a tool fails, inform the user and ask for clarification.
"""

def process_chat(messages: List[Dict[str, str]], dataset_id: str = None) -> str:
    """Process a chat interaction and return the assistant's response."""
    # Convert dicts to LangChain messages
    sys_prompt = system_prompt
    if dataset_id:
        sys_prompt += f"\n\nThe user is currently viewing the dataset with ID: '{dataset_id}'. Use this dataset ID by default when calling tools."
        
    lc_messages = [SystemMessage(content=sys_prompt)]
    for msg in messages:
        if msg["role"] == "user":
            lc_messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            lc_messages.append(AIMessage(content=msg["content"]))
            
    # Run agent
    final_state = agent.invoke({"messages": lc_messages})
    
    # Return the last AI message content
    last_msg = final_state["messages"][-1]
    content = last_msg.content
    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(block.get("text", ""))
            elif isinstance(block, str):
                text_parts.append(block)
        return "\n".join(text_parts)
    return str(content)
