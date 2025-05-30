.PHONY: help install dev test format lint typecheck run clean

help:
	@echo "Available commands:"
	@echo "  make install    - Install production dependencies"
	@echo "  make dev        - Install development dependencies"
	@echo "  make test       - Run tests"
	@echo "  make format     - Format code with black"
	@echo "  make lint       - Run flake8 linter"
	@echo "  make typecheck  - Run mypy type checker"
	@echo "  make run        - Run the application"
	@echo "  make clean      - Clean up cache files"

install:
	pip install -r requirements.txt

dev: install
	pip install pytest pytest-asyncio pytest-cov black flake8 mypy

test:
	pytest tests/ -v

format:
	black .

lint:
	flake8 .

typecheck:
	mypy .

run:
	python main.py

clean:
	find . -type d -name "__pycache__" -rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .mypy_cache htmlcov .coverage