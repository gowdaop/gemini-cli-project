# Docker Setup Instructions

## Prerequisites
- Docker Desktop installed
- Git installed
- A running Milvus instance

## Quick Start
1. Clone the repository
2. Ensure your Milvus instance is running and accessible from your host machine.
3. Run setup script: `bash config/scripts/setup.sh`
4. Update `.env` with your values. Ensure `MILVUS_HOST` and `MILVUS_PORT` are correctly set to point to your Milvus instance. If running Milvus on your local machine, `MILVUS_HOST` should be `host.docker.internal` inside the `.env` file when running the application via docker-compose.
5. Place Google Cloud credentials in `./config/credentials/`
6. Access application at `http://localhost:8000`

## Daily Development
- Start: `make up` or `docker-compose up -d`
- Stop: `make down` or `docker-compose down`
- Logs: `make logs` or `docker-compose logs -f`
- Shell: `make shell` or `docker-compose exec legal-simplifier bash`

## Troubleshooting
- If containers fail: `docker-compose down && docker-compose up --build`
- Clean restart: `make clean && make build && make up`