#!/bin/bash
set -e
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

if [ ! -f ".env" ]; then
  echo "OPENAI_API_KEY=your_openai_api_key_here" > .env
  echo "Created .env template. Edit it and re-run if needed."
fi

source ./setup.sh
mkdir -p ai_assistant/data/vision_logs

echo "Starting backend..."
python ai_assistant/backend/main.py &
BACKEND_PID=$!
sleep 2

echo "Starting frontend..."
python -m http.server 8080 --directory ai_assistant/frontend &
FRONTEND_PID=$!

trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null || true" INT TERM
echo "Backend:  http://localhost:8000"
echo "Frontend: http://localhost:8080"
wait $BACKEND_PID $FRONTEND_PID