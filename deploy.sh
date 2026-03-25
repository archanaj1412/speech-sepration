#!/bin/bash

set -e

echo "🚀 Speech Separation System Deployment Script"
echo "=============================================="

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check prerequisites
echo -e "${YELLOW}Checking prerequisites...${NC}"

if ! command -v docker &> /dev/null; then
    echo -e "${RED}Docker is not installed${NC}"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}Docker Compose is not installed${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Docker and Docker Compose found${NC}"

# Build Docker image
echo -e "${YELLOW}Building Docker image...${NC}"
docker-compose build

# Create necessary directories
echo -e "${YELLOW}Creating directories...${NC}"
mkdir -p outputs logs pretrained_models cache
chmod 755 outputs logs pretrained_models cache

# Start services
echo -e "${YELLOW}Starting services...${NC}"
docker-compose up -d

# Wait for service to be healthy
echo -e "${YELLOW}Waiting for service to be healthy...${NC}"
max_attempts=30
attempt=0

while [ $attempt -lt $max_attempts ]; do
    if curl -f http://localhost:8501/_stcore/health > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Service is healthy${NC}"
        break
    fi
    attempt=$((attempt + 1))
    echo "Attempt $attempt/$max_attempts..."
    sleep 2
done

if [ $attempt -eq $max_attempts ]; then
    echo -e "${RED}Service failed to become healthy${NC}"
    docker-compose logs
    exit 1
fi

# Display service info
echo -e "${GREEN}=============================================="
echo "✓ Deployment successful!"
echo "=============================================="
echo -e "🌐 Application URL: ${GREEN}http://localhost:8501${NC}"
echo -e "📊 Redis Cache: ${GREEN}http://localhost:6379${NC}"
echo ""
echo "Commands:"
echo "  View logs:       docker-compose logs -f"
echo "  Stop services:   docker-compose down"
echo "  Restart:         docker-compose restart"
echo ""