from openai import OpenAI

# Create client
client = OpenAI()

print("Chatbot ready. Type 'quit' to exit.\n")

# Keep the conversation history so the bot has context
messages = [
    {"role": "system", "content": "You are a patient teacher who explains concepts step by step in simple language."}
]

while True:
    # Get user input
    user_input = input("You: ")
    
    # Exit condition
    if user_input.lower() in ["quit", "exit", "bye"]:
        print("Chatbot: Goodbye!")
        break
    
    # Add user message
    messages.append({"role": "user", "content": user_input})
    
    # Get model response
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )
    
    # Extract the reply
    reply = response.choices[0].message.content
    
    # Show reply
    print("Chatbot:", reply, "\n")
    
    # Add reply to history
    messages.append({"role": "assistant", "content": reply})
