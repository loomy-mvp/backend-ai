from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

def create_chain(llm, prompt):
    """Create the chain with LangChain."""

    chain = (
        RunnablePassthrough()
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain