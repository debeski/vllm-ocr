#!/bin/bash
set -e

# Function to handle shutdown
cleanup() {
    echo "Shutting down services..."
    kill $JUPYTER_PID 2>/dev/null || true
    kill $GRADIO_PID 2>/dev/null || true
    exit 0
}

trap cleanup SIGTERM SIGINT

# Function to check HTTP endpoint
check_http_service() {
    local url=$1
    local max_attempts=10
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s --connect-timeout 5 "$url" >/dev/null 2>&1; then
            return 0
        fi
        
        # Check if process is still alive
        if [ "$url" = "http://localhost:7860" ] && ! kill -0 $GRADIO_PID 2>/dev/null; then
            echo "❌ Gradio process died while waiting"
            return 1
        fi
        
        echo "⏳ Waiting for service at $url... (attempt $attempt/$max_attempts)"
        sleep 15
        attempt=$((attempt + 1))
    done
    
    return 1
}

# Start JupyterLab in background
echo "Starting JupyterLab on port ${JUPYTER_PORT:-8888}..."
jupyter lab --ip=0.0.0.0 --port=${JUPYTER_PORT:-8888} --no-browser --allow-root --NotebookApp.token=${JUPYTER_TOKEN:-db} --NotebookApp.password='' &
JUPYTER_PID=$!

# Wait for Jupyter
if check_http_service "http://localhost:${JUPYTER_PORT:-8888}"; then
    echo "✅ JupyterLab started successfully"
else
    echo "❌ JupyterLab failed to start"
    exit 1
fi

# Start Gradio OCR app in background
echo "Starting Gradio OCR app on port 7860..."
python3 /workspace/ocr_app.py &
GRADIO_PID=$!

# Wait for Gradio
if check_http_service "http://localhost:7860"; then
    echo "✅ Gradio app started successfully"
    echo ""
    echo "🎉 Both services are up and running:"
    echo "- JupyterLab at: http://localhost:8888"
    echo "- Gradio OCR at: http://localhost:7860"
    echo ""
    echo "If running the services in -d detached mode, Use 'docker compose down' to stop it"
    echo "otherwise Press Ctrl + C twice"

else
    echo "❌ Gradio failed to start properly"
    if kill -0 $GRADIO_PID 2>/dev/null; then
        echo "⚠️ Gradio process is running but not responding - model might still be loading"
    else
        echo "❌ Gradio process has terminated"
        exit 1
    fi
fi

# Keep container running
wait -n
cleanup