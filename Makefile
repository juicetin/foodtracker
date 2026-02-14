.PHONY: dev-up dev-down

dev-up:
	docker compose -f backend/docker-compose.yml up -d

dev-down:
	docker compose -f backend/docker-compose.yml down
