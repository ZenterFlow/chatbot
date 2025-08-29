from chatbot import get_response, SYSTEM_PROMPT

messages = [{"role": "system", "content": SYSTEM_PROMPT}]

print("Chatbot ready. Type 'quit' to exit.\n")

while True:
    user_input = input("You: ")
    if user_input.lower() in ["quit", "exit", "bye"]:
        print("Chatbot: Goodbye!")
        break

    messages.append({"role": "user", "content": user_input})
    reply = get_response(messages)
    print("Chatbot:", reply, "\n")
    messages.append({"role": "assistant", "content": reply})
