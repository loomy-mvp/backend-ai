import logging
from typing import List
from langchain.schema import HumanMessage

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
        conversation_id: str,
        llm_params: dict,
        image_inputs: List[dict] = None
    ) -> str:
        """
        Generate a document based on a template and user message.
        
        Args:
            message: User's instructions or data for the document
            template: Template structure to follow
            conversation_id: Conversation ID for chat history
            llm_params: Dictionary containing provider, model, temperature, max_tokens
            image_inputs: Optional list of image inputs
            
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
                llm_params.get("max_tokens")
            )
            logger.info("[write_document] LLM initialized")
            
            # Get chat history
            chat_history = get_chat_history(conversation_id)
            logger.info(f"[write_document] Chat history length: {len(chat_history)}")
            
            # Create chain using write prompt template
            chain = create_chain(llm=llm, prompt_template=WRITE_PROMPT_TEMPLATE)
            logger.info("[write_document] Write chain created")
            
            # Build question messages
            question_messages = self._build_question_messages(message, image_inputs or [])
            
            logger.info("[write_document] LLM payload prepared")
            
            # Generate document
            response = await chain.ainvoke({
                "template": template,
                "question_messages": question_messages,
                "chat_history": chat_history
            })
            
            logger.info("[write_document] Document generated successfully")
            return response
            
        except Exception as e:
            logger.error(f"[write_document] Error generating document: {str(e)}", exc_info=True)
            raise
    
    def _build_question_messages(self, message: str, image_inputs: List[dict]) -> List[HumanMessage]:
        """Build question messages with optional image inputs."""
        content_parts: list = [
            {"type": "text", "text": f"<<<Prompt>>>\n{message}"}
        ]
        
        for image in image_inputs:
            content_parts.append({"type": "text", "text": f"[Image Attachment: {image['filename']}]"})
            content_parts.append({"type": "image_url", "image_url": {"url": image["data_url"]}})
        
        return [HumanMessage(content=content_parts)]