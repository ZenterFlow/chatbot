from openai import OpenAI
from doc_index import query_index

client = OpenAI()
SYSTEM_PROMPT = "You are a helpful assistant that answers based on provided documents."

def get_response(messages):
    # Last user question
    user_msg = messages[-1]["content"]

    # Retrieve relevant info from documents
    doc_answer = query_index(user_msg)

    # Append retrieved info to system prompt
    augmented_prompt = SYSTEM_PROMPT + "\n\nRelevant info:\n" + doc_answer

    # Send to OpenAI
    messages_with_context = [{"role": "system", "content": augmented_prompt}] + messages[1:]
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages_with_context
    )
    return response.choices[0].message.content
