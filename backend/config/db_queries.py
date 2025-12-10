"""
Database queries used across the application.
"""

# Chatbot queries
QUERY_CHAT_HISTORY = """
        SELECT * FROM messages WHERE conversation_id = %s AND status in ('generated', 'sent') ORDER BY created_at ASC;
    """