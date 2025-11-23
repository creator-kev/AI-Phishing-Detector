#!/bin/bash
echo "Testing rate limits..."
for i in {1..55}; do
  echo "Request $i:"
  curl -s -X POST http://127.0.0.1:5000/api/predict \
    -H "Content-Type: application/json" \
    -d '{"text":"https://google.com"}' | jq '.verdict // .error'
  sleep 1
done
