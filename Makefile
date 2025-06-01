.PHONY: install install-dev test test-cov lint format check-format check-types clean clean-build clean-pyc clean-test help

# Variables
PYTHON = python
PIP = pip
POETRY = poetry
PYTEST = $(POETRY) run pytest
PYTEST_COV = $(PYTEST) --cov=quantalogic_pythonbox --cov-report=term-missing
BLACK = $(POETRY) run black
ISORT = $(POETRY) run isort
MYPY = $(POETRY) run mypy
PYLINT = $(POETRY) run pylint

# Default target
help:
	@echo "Please use 'make <target>' where <target> is one of:"
	@echo "  install         Install the package and its dependencies"
	@echo "  install-dev     Install development dependencies"
	@echo "  test            Run tests"
	@echo "  test-cov        Run tests with coverage report"
	@echo "  lint            Run all linters"
	@echo "  format          Format code with black and isort"
	@echo "  check-format    Check code formatting without making changes"
	@echo "  check-types     Run type checking with mypy"
	@echo "  clean           Remove all build, test, coverage and Python artifacts"
	@echo "  clean-build     Remove build artifacts"
	@echo "  clean-pyc       Remove Python file artifacts"
	@echo "  clean-test      Remove test and coverage artifacts"

# Installation
install:
	$(POETRY) install --only main

install-dev:
	$(POETRY) install

# Testing
test:
	$(PYTEST) -v tests/

test-cov:
	$(PYTEST_COV) tests/

# Linting and formatting
lint:
	@echo "Running black..."
	$(BLACK) --check --diff .
	@echo "\nRunning isort..."
	$(ISORT) --check-only --diff .
	@echo "\nRunning mypy..."
	$(MYPY) quantalogic_pythonbox/
	@echo "\nRunning pylint..."
	$(PYLINT) quantalogic_pythonbox/

format:
	$(BLACK) .
	$(ISORT) .

check-format:
	$(BLACK) --check .
	$(ISORT) --check-only .

check-types:
	$(MYPY) quantalogic_pythonbox/

# Cleanup
clean: clean-build clean-pyc clean-test

clean-build:
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -fr {} +

clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test:
	rm -fr .pytest_cache/
	rm -fr .mypy_cache/
	rm -f .coverage
	rm -fr htmlcov/
