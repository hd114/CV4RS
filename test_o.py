from openai import OpenAI
import os

# Setze den API-Schlüssel aus der Umgebungsvariable
skey = os.getenv("OPENAI_API_KEY")
## Set the API key
client = OpenAI(api_key=skey)

MODEL=("gpt-4o")

completion = client.chat.completions.create(
  model=MODEL,
  messages=[
    {"role": "system", "content": "You are a helpful assistant that helps me with my math homework!"},
    {"role": "user", "content": "Hello! Could you solve 20 x 5?"}
  ]
)
print("Assistant: " + completion.choices[0].message.content)
