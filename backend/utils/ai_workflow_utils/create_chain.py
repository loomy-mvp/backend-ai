from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

def create_chain(llm, prompt_template):
    """Create the chain with LangChain."""

    chain = (
        RunnablePassthrough()
        | prompt_template
        | llm
        | StrOutputParser()
    )
    
    return chain