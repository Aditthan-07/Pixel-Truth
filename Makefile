.PHONY: help test run docker-build docker-up clean

help:
	@echo "PixelTruth Development Tasks:"
	@echo "  make test          - Run full unit test suite"
	@echo "  make run           - Run Flask local development server"
	@echo "  make docker-build  - Build Docker containers"
	@echo "  make docker-up     - Launch Docker containers"
	@echo "  make clean         - Remove bytecode and cache artifacts"

test:
	python -m unittest discover -s tests

run:
	python run.py

docker-build:
	docker compose build

docker-up:
	docker compose up

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
