#!/bin/bash
# Local training script - runs everything in one go

set -e

echo "=========================================="
echo "NegotiateEnv Local Training"
echo "=========================================="
echo ""

# Check if server is already running
if lsof -Pi :7860 -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo "✓ Server already running on port 7860"
else
    echo "Starting environment server..."
    python -m uvicorn negotiate_env.server.app:app --host 0.0.0.0 --port 7860 --log-level error > /tmp/negotiate_server.log 2>&1 &
    SERVER_PID=$!
    echo "Server PID: $SERVER_PID"
    
    # Wait for server to be ready
    echo "Waiting for server to start..."
    for i in {1..30}; do
        if curl -s http://localhost:7860/health > /dev/null 2>&1; then
            echo "✓ Server ready!"
            break
        fi
        sleep 1
        if [ $i -eq 30 ]; then
            echo "✗ Server failed to start. Check logs:"
            tail -20 /tmp/negotiate_server.log
            exit 1
        fi
    done
fi

echo ""
echo "Testing server..."
python test_server.py

echo ""
echo "=========================================="
echo "Starting Training"
echo "=========================================="
echo "Configuration:"
echo "  - Episodes: 1000"
echo "  - Max turns: 50"
echo "  - Model: Qwen/Qwen2.5-1.5B-Instruct"
echo "  - Output: negotiate-long-horizon-output/"
echo ""

python train_negotiate_unsloth.py \
    --env-url http://localhost:7860 \
    --model-id Qwen/Qwen2.5-1.5B-Instruct \
    --output-dir negotiate-long-horizon-output \
    --num-episodes 1000 \
    --max-turns 50

echo ""
echo "=========================================="
echo "Training Complete!"
echo "=========================================="
echo "Model saved to: negotiate-long-horizon-output/"
echo ""
echo "To stop the server:"
echo "  kill $SERVER_PID"
