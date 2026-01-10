import logging

from backend.config.chatbot_config import WRITER_PROVIDER_THINKING_KWARGS
from backend.config.prompts import WRITE_PROMPT_TEMPLATE
from backend.utils.ai_workflow_utils.get_llm import get_llm
from backend.utils.ai_workflow_utils.create_chain import create_chain
from backend.utils.ai_workflow_utils.get_chat_history import get_chat_history

logger = logging.getLogger(__name__)


class Writer:
    def __init__(self):
        pass
    
    async def write_document(
        self,
        message: str,
        template: str,
        requirements: str,
        conversation_id: str,
        llm_params: dict
    ) -> str:
        """
        Generate a document based on a template and user message.
        
        Args:
            message: User's instructions or data for the document
            template: Template structure to follow
            requirements: JSON formatted string with template field values
            conversation_id: Conversation ID for chat history
            llm_params: Dictionary containing provider, model, temperature, max_tokens
            
        Returns:
            Generated document as a string
        """
        try:
            logger.info("[write_document] Starting document generation")
            
            # Initialize LLM
            llm = get_llm(
                llm_params.get("provider"),
                llm_params.get("model"),
                llm_params.get("temperature"),
                llm_params.get("max_tokens"),
                provider_thinking_kwargs=WRITER_PROVIDER_THINKING_KWARGS,
            )
            logger.info("[write_document] LLM initialized")
            
            # Get chat history
            chat_history = get_chat_history(conversation_id)
            logger.info(f"[write_document] Chat history length: {len(chat_history)}")
            
            # Create chain using write prompt template
            chain = create_chain(llm=llm, prompt_template=WRITE_PROMPT_TEMPLATE)
            logger.info("[write_document] Write chain created")
            
            logger.info("[write_document] LLM payload prepared")
            
            # Generate document
            response = await chain.ainvoke({
                "template": template,
                "requirements": requirements,
                "message": message,
                "chat_history": chat_history
            })
            
            logger.info("[write_document] Document generated successfully")
            return response
            
        except Exception as e:
            logger.error(f"[write_document] Error generating document: {str(e)}", exc_info=True)
            raise