.PHONY: dev build start stop clean backend frontend db logs

# Development
dev:
	docker compose up --build

dev-backend:
	cd backend && uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

dev-frontend:
	cd frontend && npm install && npm run dev

# Production
build:
	docker compose -f docker-compose.prod.yml build

start:
	docker compose -f docker-compose.prod.yml up -d

stop:
	docker compose down

# Individual services
backend:
	docker compose up backend -d

frontend:
	docker compose up frontend -d

db:
	docker compose up db -d

# Utilities
logs:
	docker compose logs -f

logs-backend:
	docker compose logs -f backend

logs-frontend:
	docker compose logs -f frontend

clean:
	docker compose down -v
	rm -rf frontend/node_modules
	rm -rf frontend/dist
	rm -rf backend/__pycache__

# Setup
setup-frontend:
	cd frontend && npm install

setup-backend:
	cd backend && pip install -r requirements.txt

# Database
db-init:
	docker compose exec backend python -c "from app.db.database import init_db; import asyncio; asyncio.run(init_db())"

db-shell:
	docker compose exec db psql -U app -d app
