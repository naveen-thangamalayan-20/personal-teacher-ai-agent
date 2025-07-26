import pprint

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from typing import Annotated, Sequence
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage

from langgraph.graph.message import add_messages

from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition

import config
from retriever import retrieve

llm = ChatOllama(model=config.OLLAMA_MODEL, base_url="http://127.0.0.1:11434")


def agent(state):
    """
    Invokes the agent model to generate a response based on the current state. Given
    the question, it will decide to retrieve using the retriever tool, or simply end.

    Args:
        state (messages): The current state

    Returns:
        dict: The updated state with the agent response appended to messages
    """
    print("---CALL AGENT---")
    messages = state["messages"]
    print(messages)
    model = llm
    model = model.bind_tools([retrieve])
    response = model.invoke(messages)
    print(response)
    print("######")
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


def generate(state):
    """
    Generate answer

    Args:
        state (messages): The current state

    Returns:
         dict: The updated state with re-phrased question
    """
    print("---GENERATE---")
    messages = state["messages"]
    question = messages[0].content
    last_message = messages[-1]

    docs = last_message.content
    template = """Assume you are a personal tutor. 
    Based on the below context can you generate a quiz questions with 3 options to evaluate my understanding. 
    There should be only one answer to the quiz question and also provide reason why it is the correct answer:
    {context}


    """
    prompt = ChatPromptTemplate.from_template(template)

    # prompt = prompt_template.invoke({"context": docs})
    # Prompt
    # prompt = hub.pull("rlm/rag-prompt")

    # LLM
    # llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, streaming=True)

    # Post-processing
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Chain
    rag_chain = prompt | llm | StrOutputParser()

    # Run
    response = rag_chain.invoke({"context": docs, "question": question})
    print("#########MEssages######")
    print(messages)
    print("#########End MEssages######")
    return {"messages": [response]}


class AgentState(TypedDict):
    # The add_messages function defines how an update should be processed
    # Default is to replace. add_messages says "append"
    messages: Annotated[Sequence[BaseMessage], add_messages]

def user_answer(state):
    messages = state["messages"]
    context = messages[-2].content
    question = messages[-1].content

    answer =  input("Enter your answer")
    template = """Assume you are a personal tutor. 
    Based on the below context:
    {context}
    You have generated the below question:
    {question}
    For the above question, Can you evaluate the answer option entered by the user: {answer} 
    If the user has answered correctly appreciate him if it is a wrong answer then only provide a constructive feedback with answer
    


    """
    prompt = ChatPromptTemplate.from_template(template)
    rag_chain = prompt | llm | StrOutputParser()

    # Run
    response = rag_chain.invoke({"context": context, "question": question, "answer": answer})
    return {"messages": [response]}
    # messages = state["messages"]
    # question = messages[0].content
    # last_message = messages[-1]

def create_graph():
    workflow = StateGraph(AgentState)

    # Define the nodes we will cycle between
    workflow.add_node("agent", agent)  # agent
    retrieve_node = ToolNode([retrieve])
    workflow.add_node("retrieve", retrieve_node)  # retrieval
    # workflow.add_node("rewrite", rewrite)  # Re-writing the question
    workflow.add_node(
        "generate", generate
    )  # Generating a response after we know the documents are relevant
    workflow.add_node(
        "user_answer", user_answer
    )  # Generating a response after we know the documents are relevant
    # Call agent node to decide to retrieve or not
    workflow.add_edge(START, "agent")

    # Decide whether to retrieve
    workflow.add_conditional_edges(
        "agent",
        # Assess agent decision
        tools_condition,
        {
            # Translate the condition outputs to nodes in our graph
            "tools": "retrieve",
            END: END,
        },
    )

    # Edges taken after the `action` node is called.
    # workflow.add_conditional_edges(
    #     "retrieve",
    #     # Assess agent decision
    #     grade_documents,
    # )v
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", "user_answer")
    workflow.add_edge("user_answer", END)
    return workflow.compile()
    # workflow.add_edge("rewrite", "agent")



def run():
    graph = create_graph()
    inputs = {
        "messages": [
            ("user", "Architecture of Madurai?"),
        ]
    }
    for output in graph.stream(inputs):
        for key, value in output.items():
            pprint.pprint(f"Output from node '{key}':")
            pprint.pprint("---")
            pprint.pprint(value, indent=2, width=80, depth=None)
        pprint.pprint("\n---\n")


run()
