.PHONY: build up down logs shell test clean

# Build the Docker image
build:
	docker-compose build

# Start the services
up:
	docker-compose up -d

# Stop the services
down:
	docker-compose down

# View logs
logs:
	docker-compose logs -f

# Access application shell
shell:
	docker-compose exec legal-simplifier bash

# Run tests
test:
	docker-compose exec legal-simplifier pytest

# Clean up everything
clean:
	docker-compose down -v
	docker system prune -f
git status
git stash push -m "wip before pull --rebase"
git pull --rebase origin
git stash pop


git commit -m "Your commit message here"

git push origin main


python scripts/generate_openapi.py \
  --app-dir src/legal-document-simplifier \
  --app-path src.backend.main:app \
  --out docs/openapi.json

docker compose restart legal-simplifier

docker compose up --build legal-simplifier

docker compose logs -f legal-simplifier

pytest -v /app/src/legal-document-simplifier/src/backend/tests/

docker compose down && docker compose up -d

# one-shot build & run
docker compose build --no-cache
docker compose up -d

# local hot-reload for React
docker compose --profile dev up frontend-dev

legal-doc-analyzer-2025-secure-key-f47d4a2c

docker compose logs -f frontend-dev  