#!/bin/bash

set -e

echo "🚀 Deploying with Docker Compose"
echo "=================================="

# Create directories
mkdir -p outputs logs pretrained_models cache

# Build
echo "🔨 Building Docker image..."
docker-compose build

# Start
echo "▶️  Starting services..."
docker-compose up -d

# Wait for health
echo "⏳ Waiting for service to be healthy..."
for i in {1..60}; do
    if curl -f http://localhost:8501/_stcore/health > /dev/null 2>&1; then
        echo "✅ Service is healthy!"
        break
    fi
    echo "Attempt $i/60..."
    sleep 2
done

echo ""
echo "✅ Deployment complete!"
echo "======================="
echo "🌐 Access at: http://localhost:8501"
echo ""
echo "Commands:"
echo "  docker-compose logs -f"
echo "  docker-compose down"