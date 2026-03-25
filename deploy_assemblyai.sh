#!/bin/bash

set -e

echo "🚀 Deploying Speech Separation + AssemblyAI Real-Time"
echo "======================================================="

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Check Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not installed"
    exit 1
fi

echo -e "${YELLOW}Creating directories...${NC}"
mkdir -p outputs logs pretrained_models cache

echo -e "${YELLOW}Building Docker image...${NC}"
docker-compose -f docker-compose.assemblyai.yml build

echo -e "${YELLOW}Starting services...${NC}"
docker-compose -f docker-compose.assemblyai.yml up -d

echo -e "${YELLOW}Waiting for services to be ready...${NC}"
for i in {1..30}; do
    if curl -f http://localhost:8501/_stcore/health > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Service is healthy!${NC}"
        break
    fi
    echo "Waiting... ($i/30)"
    sleep 2
done

echo -e "${GREEN}======================================================="
echo "✅ Deployment successful!"
echo "======================================================="
echo -e "🌐 Access at: ${GREEN}http://localhost:8501${NC}"
echo ""
echo "📚 Setup Instructions:"
echo "1. Get AssemblyAI API key: https://www.assemblyai.com"
echo "2. Enter API key in the sidebar"
echo "3. Click '🔴 Real-Time Transcription (Live)'"
echo "4. Click 'Start Recording' and speak into your microphone"
echo ""
echo "Commands:"
echo "  docker-compose -f docker-compose.assemblyai.yml logs -f"
echo "  docker-compose -f docker-compose.assemblyai.yml down"