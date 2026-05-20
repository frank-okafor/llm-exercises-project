from typing import Dict, List
from openai import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# System prompt for NASA mission expert
SYSTEM_PROMPT = """You are a NASA mission archive assistant and space exploration expert. Your role is to provide accurate, well-sourced information about NASA space missions based on the retrieved context provided.

Guidelines:
1. Use the provided retrieved context as your primary evidence base.
2. Do NOT invent facts that are not supported by the context.
3. If the context is insufficient to answer the question, clearly state what information is missing.
4. Prefer concise, technically correct answers.
5. When practical, mention the mission and source snippets that support your answer.
6. If you cite specific information, reference which source it came from (e.g., "According to Source 1...").
7. For technical details, be precise and use proper terminology.
8. If asked about missions not covered in the context, acknowledge the limitation.

Remember: Accuracy and honesty are paramount. It's better to admit uncertainty than to provide incorrect information."""


def build_system_prompt() -> str:
    """Build the system prompt for the NASA assistant"""
    return SYSTEM_PROMPT


def generate_response(openai_key: str, user_message: str, context: str, 
                     conversation_history: List[Dict], model: str = "gpt-4o-mini") -> str:
    """Generate response using OpenAI with context

    Args:
        openai_key: OpenAI API key
        user_message: The user's question
        context: Retrieved context from the RAG system
        conversation_history: Previous conversation turns
        model: OpenAI model to use

    Returns:
        Generated response string
    """

    # Define system prompt
    system_prompt = build_system_prompt()

    # Create OpenAI client
    client = OpenAI(
        api_key=openai_key,
        base_url=os.getenv("OPENAI_BASE_URL", "https://openai.vocareum.com/v1")
    )

    # Build messages list
    messages = []

    # Add system prompt
    messages.append({
        "role": "system",
        "content": system_prompt
    })

    # Add chat history (limit to last 10 exchanges to conserve tokens)
    history_limit = 10
    recent_history = conversation_history[-history_limit:] if len(conversation_history) > history_limit else conversation_history

    for msg in recent_history:
        messages.append({
            "role": msg.get("role", "user"),
            "content": msg.get("content", "")
        })

    # Set context in messages - combine context and user question
    if context:
        user_content = f"""Based on the following retrieved context, please answer my question.

{context}

=== User Question ===
{user_message}

Please provide a helpful and accurate answer based primarily on the retrieved context above. If the context doesn't contain enough information to fully answer the question, please indicate what information is missing."""
    else:
        user_content = f"""{user_message}

Note: No relevant context was retrieved for this question. Please indicate if you need more specific information to provide an accurate answer about NASA missions."""

    messages.append({
        "role": "user",
        "content": user_content
    })

    # Send request to OpenAI
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7,
            max_tokens=1000
        )

        # Return response
        content = response.choices[0].message.content
        return content if content else "No response generated."

    except Exception as e:
        return f"Error generating response: {str(e)}"


def append_history(history: List[Dict], role: str, content: str) -> List[Dict]:
    """Append a message to conversation history

    Args:
        history: Existing conversation history
        role: Role of the message (user or assistant)
        content: Content of the message

    Returns:
        Updated conversation history
    """
    history.append({
        "role": role,
        "content": content
    })
    return history


def get_conversation_summary(history: List[Dict], max_length: int = 500) -> str:
    """Get a summary of the conversation history

    Args:
        history: Conversation history
        max_length: Maximum length of summary

    Returns:
        Summary string
    """
    if not history:
        return "No previous conversation."

    summary_parts = []
    for msg in history[-5:]:  # Last 5 messages
        role = msg.get("role", "unknown").capitalize()
        content = msg.get("content", "")[:100]  # First 100 chars
        summary_parts.append(f"{role}: {content}...")

    summary = "\n".join(summary_parts)
    if len(summary) > max_length:
        summary = summary[:max_length] + "..."

    return summary
