from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

def create_chain(llm, prompt_template):
    """Create the chain with LangChain."""

    chain = (
        RunnablePassthrough()
        | prompt_template
        | llm
        | StrOutputParser()
    )
    
    return chain