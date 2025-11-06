import requests, json

response = requests.post(
    "http://localhost:1234/v1/chat/completions",  # change 1234 if needed
    headers={"Content-Type": "application/json"},
    data=json.dumps({
        "model": "codellama-7b-instruct",  # name doesn’t have to match exactly, but keep it descriptive
        "messages": [{"role": "user", "content": "Say hello"}]
    })
)

print(response.json())