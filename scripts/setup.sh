#!/bin/bash

# Setup script for Legal Document Simplifier

echo "Setting up Legal Document Simplifier development environment..."

# Copy environment file
if [ ! -f .env ]; then
    cp .env.example .env
    echo "Created .env file - please update with your actual values"
fi

# Create credentials directory
mkdir -p config/credentials
echo "Place your Google Cloud service account JSON file in ./config/credentials/"

# Build and start containers
docker-compose build
docker-compose up -d

echo "Setup complete! Access the application at http://localhost:8000"
echo "Don't forget to:"
echo "1. Update .env with your actual values"
echo "2. Place Google Cloud credentials in ./config/credentials/"
echo "3. Run 'docker-compose logs' to check if everything is working"
