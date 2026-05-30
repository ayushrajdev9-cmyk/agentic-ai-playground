.PHONY: install install-dev lint format typecheck test clean

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"
	pre-commit install

lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/

typecheck:
	mypy src/

test:
	pytest --cov=src/ -v

test-coverage:
	pytest --cov=src/ --cov-report=html --cov-report=term

clean:
	rm -rf build/ dist/ *.egg-info/ .pytest_cache/ .mypy_cache/
	rm -rf htmlcov/ .coverage coverage/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

all: lint typecheck test
