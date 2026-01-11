from langgraph.graph import StateGraph
from graph.state import RAGState
from agents.agent import retrieve_node
from agents.agent import generate_node


def build_multirag_graph():
    graph = StateGraph(RAGState)
    
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    
    return graph.compile()
