from openai import OpenAI

# Create client
client = OpenAI()

# Send a simple prompt
response = client.chat.completions.create(
    model="gpt-4o-mini",  # smaller, fast, cheaper model
    messages=[
        {"role": "system", "content": "You are a friendly assistant."},
        {"role": "user", "content": "Explain VO2 Max as if I were 10 years old."}
    ]
)

# Print the model's reply
print(response.choices[0].message.content)