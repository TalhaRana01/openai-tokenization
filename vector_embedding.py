from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI()


text = "Eiffel Tower is in Paris and is a famous landmark.it is 324 meters tall."


response = client.embeddings.create(
    input=text,
    model="text-embedding-3-small",
    
)


print("Embedding Vector Length:", response.data[0].embedding) 
