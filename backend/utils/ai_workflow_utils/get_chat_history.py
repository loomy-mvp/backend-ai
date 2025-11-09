# DB connection
from backend.utils.db_utils import DBUtils
from backend.config.db_queries import QUERY_CHAT_HISTORY
from typing import List, Dict, Any

def get_chat_history(conversation_id: str) -> List[Dict[str, Any]]:
    """Get conversation memory for a conversation from the database"""
    db_results = DBUtils.execute_query(
        QUERY_CHAT_HISTORY,
        (conversation_id,) # tuple because the inner method is: cursor.execute(query, params)
    )

    messages = []
    for msg in db_results:
        if msg[6] != 'error':
            message_content = {
                "sender": msg[2],
                "content": msg[3],
                "metadata": msg[4]
                # "created_at": msg[5]
            }
            messages.append(message_content)
    
    return messages