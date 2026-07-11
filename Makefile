.PHONY: install install-dev download prepare-data train evaluate visualize run test lint clean

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

download:
	avavs download --config configs/default.yaml

prepare-data:
	avavs prepare-data --config configs/default.yaml

train:
	avavs train --config configs/default.yaml

evaluate:
	avavs evaluate --config configs/default.yaml

visualize:
	avavs visualize --config configs/default.yaml

run:
	avavs run --config configs/default.yaml

quick-test:
	avavs run --config configs/quick_test.yaml

test:
	pytest tests/ -v --cov=avavs --cov-report=term-missing

lint:
	ruff check src/ tests/ scripts/
	black --check src/ tests/ scripts/

clean:
	rm -rf outputs/ outputs_quick_test/ avmit_cache/ *.egg-info build/ dist/
	find . -type d -name __pycache__ -exec rm -rf {} +
