from typing import Any, Dict
import json
import re
from pydantic import BaseModel
from backend.utils.ai_workflow_utils.get_llm import get_llm
from backend.utils.ai_workflow_utils.create_chain import create_chain
from backend.config.chatbot_config import RETRIEVAL_JUDGE_CONFIG
from backend.config.prompts import RETRIEVAL_JUDGE_PROMPT_TEMPLATE

class RetrievalDecision(BaseModel):
    """Pydantic model that enforces the 'retrieve' field as a boolean.

    This model is used to validate/coerce the LLM's output so the function
    always returns a proper Python bool.
    """
    retrieve: bool

class RetrievalJudge:
    
    def __init__(self):
        self.llm = get_llm(
            provider=RETRIEVAL_JUDGE_CONFIG["provider"],
            model=RETRIEVAL_JUDGE_CONFIG["model"],
            max_tokens=RETRIEVAL_JUDGE_CONFIG["max_tokens"],
            temperature=RETRIEVAL_JUDGE_CONFIG["temperature"]
        )

    def _parse_llm_response(self, raw: Any) -> Dict[str, Any]:
        """Try to convert various LLM outputs into a dict with a 'retrieve' key.

        The LLM may return a dict-like object, a JSON string, or a freeform
        string. This helper attempts JSON parsing first, then a regex fallback to
        extract True/False values.
        """
        # If it's already a mapping-like object, try using it directly.
        if isinstance(raw, dict):
            return raw

        # If it's bytes, decode to str
        if isinstance(raw, bytes):
            raw = raw.decode(errors="ignore")

        if isinstance(raw, str):
            # Try JSON first
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass

            # Fallback: look for a retrieve: true/false pattern
            m = re.search(r'"?retrieve"?\s*[:=]\s*"?(True|False|true|false)"?', raw)
            if m:
                return {"retrieve": m.group(1)}

        # As a last resort, return an empty dict so pydantic will raise a clear error
        return {}


    def judge_retrieval(self, chat_history: str, message: str) -> bool:
        """Use an LLM to judge if the retrieved context answers the query.

        Returns:
            bool: True if another retrieval stage is recommended, False otherwise.

        The LLM's raw response is validated and coerced through the
        RetrievalDecision pydantic model to guarantee a boolean return value.
        """

        prompt = RETRIEVAL_JUDGE_PROMPT_TEMPLATE.format(
            message=message,
            chat_history=chat_history
        )

        chain = create_chain(llm=self.llm, prompt=prompt)
        raw_response = chain.invoke({
            "message": message,
            "chat_history": chat_history
        })

        parsed = self._parse_llm_response(raw_response)

        # Use pydantic to coerce/validate the value to a bool. This will raise a ValidationError if the result cannot be interpreted as a boolean.
        decision = RetrievalDecision.model_validate(parsed)
        return bool(decision.retrieve)