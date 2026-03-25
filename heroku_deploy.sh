#!/bin/bash

set -e

echo "🚀 Deploying to Heroku"
echo "====================="

# Check prerequisites
if ! command -v heroku &> /dev/null; then
    echo "❌ Heroku CLI not installed. Installing..."
    curl https://cli-assets.heroku.com/install.sh | sh
fi

# Variables
APP_NAME=${1:-speech-separation-ai}
API_KEY=${2:-""}

# Login to Heroku
echo "🔑 Logging into Heroku..."
heroku login -i

# Create app if doesn't exist
if ! heroku apps:info --app=$APP_NAME > /dev/null 2>&1; then
    echo "📦 Creating Heroku app: $APP_NAME"
    heroku create $APP_NAME --region=us
else
    echo "✅ App $APP_NAME exists"
fi

# Set buildpack
echo "🔨 Setting buildpack..."
heroku buildpacks:set heroku/python --app=$APP_NAME

# Set environment variables
if [ ! -z "$API_KEY" ]; then
    echo "🔑 Setting AssemblyAI API Key..."
    heroku config:set ASSEMBLYAI_API_KEY=$API_KEY --app=$APP_NAME
fi

# Additional configs
heroku config:set PYTHONUNBUFFERED=1 --app=$APP_NAME
heroku config:set STREAMLIT_SERVER_MAXUPLOADSIZE=500 --app=$APP_NAME

# Deploy
echo "📤 Deploying code..."
git push heroku main

# Display app info
echo ""
echo "✅ Deployment successful!"
echo "========================="
echo "🌐 App URL: https://$APP_NAME.herokuapp.com"
echo ""
echo "Useful commands:"
echo "  heroku logs --tail --app=$APP_NAME"
echo "  heroku open --app=$APP_NAME"
echo "  heroku config --app=$APP_NAME"