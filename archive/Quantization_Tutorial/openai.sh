unset http_proxy
unset https_proxy

curl http://localhost:8000/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": "Hello."
            }
        ],
        "max_tokens": 100,
        "temperature": 0.6
    }'