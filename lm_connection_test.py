import requests
import json

url = "http://localhost:1234/v1/chat/completions"

payload = {
    "model": "codellama-7b-instruct",
    "messages": [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "Write a simple SQL query to get all rows from a table named sales."}
    ],
    "temperature": 0.7
}

headers = {"Content-Type": "application/json"}

response = requests.post(url, headers=headers, data=json.dumps(payload))
print(response.json())