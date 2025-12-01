#!/bin/bash
set -e

# Function to handle shutdown
cleanup() {
    echo "Shutting down services..."
    kill $JUPYTER_PID 2>/dev/null || true
    kill $GRADIO_PID 2>/dev/null || true
    kill $FASTAPI_PID 2>/dev/null || true
    exit 0
}

trap cleanup SIGTERM SIGINT

# Function to check HTTP endpoint
check_http_service() {
    local url=$1
    local max_attempts=20
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s --connect-timeout 5 "$url" >/dev/null 2>&1; then
            return 0
        fi
        
        # Check if processes died
        if [ "$url" = "http://localhost:7860" ] && ! kill -0 $GRADIO_PID 2>/dev/null; then
            echo "❌ Gradio process died while waiting"
            return 1
        fi

        if [ "$url" = "http://localhost:8000" ] && ! kill -0 $FASTAPI_PID 2>/dev/null; then
            echo "❌ FastAPI process died while waiting"
            return 1
        fi
        
        echo "⏳ Waiting for service at $url... (attempt $attempt/$max_attempts)"
        sleep 10
        attempt=$((attempt + 1))
    done
    
    return 1
}

##############################################
# Start JupyterLab
##############################################

echo "Starting JupyterLab on port ${JUPYTER_PORT:-8888}..."
jupyter lab --ip=0.0.0.0 --port=${JUPYTER_PORT:-8888} --no-browser --allow-root \
    --NotebookApp.token=${JUPYTER_TOKEN:-db} --NotebookApp.password='' &

JUPYTER_PID=$!

# Wait for Jupyter
if check_http_service "http://localhost:${JUPYTER_PORT:-8888}"; then
    echo "✅ JupyterLab started successfully"
else
    echo "❌ JupyterLab failed to start"
    exit 1
fi

##############################################
# Start Gradio OCR UI
##############################################

echo "Starting Gradio OCR app on port 7860..."
python3 /workspace/ocr_app.py &
GRADIO_PID=$!

if check_http_service "http://localhost:7860"; then
    echo "✅ Gradio app started successfully"
else
    echo "❌ Gradio failed to start properly"
fi

##############################################
# Start FastAPI OCR API
##############################################

echo "Starting FastAPI OCR API on port 8000..."
uvicorn ocr_api:app --host 0.0.0.0 --port 8000 &
FASTAPI_PID=$!

if check_http_service "http://localhost:8000"; then
    echo "✅ FastAPI OCR API started successfully"
else
    echo "❌ FastAPI failed to start properly"
fi

##############################################
# Status summary
##############################################

echo ""
echo "🎉 All services running:"
echo "- JupyterLab: http://localhost:${JUPYTER_PORT:-8888}"
echo "- Gradio UI:  http://localhost:7860"
echo "- FastAPI OCR: http://localhost:8000/api/ocr"
echo ""
echo "Use Ctrl+C to stop (or docker compose down)"
echo ""

##############################################
# Keep container alive
##############################################
wait -n
cleanup
