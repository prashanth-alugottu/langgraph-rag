from langchain_openai import ChatOpenAI
from graph.state import RAGState
from services.uploading_service import get_vector_db
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
import os

# llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
)


def retrieve_node(state):
    query = state["query"]
    print("\n🧲 Retrieval Agent received query:", query)
    db = get_vector_db()
    docs = db.similarity_search(query, k=5)
    print(f"🧲 Retrieved {len(docs)} documents.")
    for doc in docs:
        print("----- Document chunk:", doc.page_content[:100].replace("\n"," "), "...")
    return {"retrieved_docs": docs}



def generate_node(state: RAGState):
    docs = state["retrieved_docs"]

    if not docs:
        return {
            "answer": "Not found in the provided documents."
        }

    context = "\n\n".join(
        [d.page_content for d in docs]
    )

    prompt = f"""
        You are a factual assistant.

        Answer ONLY using the context below.
        If the answer is not present, say:
        "Not found in the provided documents."

        Context:
        {context}

        Question:
        {state['query']}
        """

    response = llm.invoke(prompt)

    return {
        "answer": response.content,
         "sources": [docs]
    }
 

 
# def rewrite_query_node(state: RAGState):
#     original_query = state["query"]
#     print("\n📝 Original Query:", original_query)
#     prompt = f"""
#         You are a query rewriting assistant.

#         Rewrite the user's query to be more specific
#         for retrieving relevant documents from a vector database.

#         Original Query:
#         {original_query}

#         Rewritten Query:
#         """
#     response = llm.invoke(prompt)
#     print("\n📝 Query Rewriting Response:", response.content)
#     rewritten_query = response.content.strip()
#     print("\n✍️ Rewritten Query:", rewritten_query)
#     return {"rewritten_query": rewritten_query}


# reranker = CrossEncoder(
#     "cross-encoder/ms-marco-MiniLM-L-6-v2"
# )

# def rerank_node(state: RAGState):
#     docs = state["retrieved_docs"]
#     query = state["query"]

#     if not docs:
#         return {"reranked_docs": []}

#     pairs = [(query, d.page_content) for d in docs]
#     print("\n🔎 Reranking pairs : ",pairs)
#     scores = reranker.predict(pairs)
#     print("\n🔎 Reranking scores : ",scores)

#     ranked = sorted(
#         zip(docs, scores),
#         key=lambda x: x[1],
#         reverse=True
#     )
#     print("\n🔎 Reranked documents and scores : ",ranked)

#     # keep only strong evidence
#     top_docs = [
#         doc for doc, score in ranked[:5]
#         if score > 0.2
#     ]
#     print("Top Docs : ",top_docs)

#     for doc in top_docs:
#         print("----- Top Document chunk:", doc.page_content[:100].replace("\n"," "), "...")

#     return {"reranked_docs": top_docs}
   
# def faithfulness_node(state: RAGState):
#     if not state["grounded"]:
#         return state
#     docs = state["reranked_docs"]
#     context = "\n\n".join(d.page_content for d in docs)
#     print("\n🔍 Faithfulness check context:", context)
#     prompt = f"""
#             Check if the answer is fully supported by the context.

#             Context:
#             {context}

#             Answer:
#             {state['answer']}

#             Respond only YES or NO.
#             """
#     verdict = llm.invoke(prompt).content.strip()

#     if verdict == "NO":
#         return {
#             "answer": "Not found in the provided documents.",
#             "grounded": False
#         }

#     return state
