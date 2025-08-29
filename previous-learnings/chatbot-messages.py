from openai import OpenAI

# OpenAI client
client = OpenAI()

# System prompt for the chatbot
SYSTEM_PROMPT = "You are a friendly assistant who explains things simply."

def get_response(messages):
    """
    Takes a list of messages, calls OpenAI API, returns assistant reply.
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )
    return response.choices[0].message.content
