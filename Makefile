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
