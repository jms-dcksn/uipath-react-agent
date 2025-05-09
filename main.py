from langchain_openai import ChatOpenAI
from langgraph.graph import START, StateGraph, END
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel
from langchain_community.tools.tavily_search import TavilySearchResults


class GraphInput(BaseModel):
    query: str


class GraphOutput(BaseModel):
    response: str


model = ChatOpenAI(model="gpt-4o")
tavily_tool = TavilySearchResults(max_results=5)
agent = create_react_agent(model, tools=[tavily_tool], prompt="You are a helpful assistant. Use your web search tool.")

async def agent_node(state: GraphInput) -> GraphOutput:
    new_state = MessagesState(messages=[{"role": "user", "content": state.query}])

    result = await agent.ainvoke(new_state)

    return GraphOutput(response=result["messages"][-1].content)

# Build the state graph
builder = StateGraph(input=GraphInput, output=GraphOutput)
builder.add_node("agent", agent_node)

builder.add_edge(START, "agent")
builder.add_edge("agent", END)

# Compile the graph
graph = builder.compile()

