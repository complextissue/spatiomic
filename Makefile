.DEFAULT_GOAL := test

create-venv:
	uv venv '.venv' --python 3.12
	echo "Don't forget to activate with 'source .venv/bin/activate'"

#
# Install requirements
#
install:
	uv sync

install-dev:
	uv sync --extra cellpose --dev
	python -m certifi
	pre-commit install --hook-type pre-commit --hook-type pre-push

#
# Checks & package upload
#
check: install-dev
	uv run ruff check --fix --exit-non-zero-on-fix spatiomic
	uv run ruff format --check spatiomic
	uv run mypy -p spatiomic
	uv run bandit -ll --recursive spatiomic

#
# Testing
#
unittest:
	uv run coverage run -m pytest --maxfail=10 -m "not gpu"

unittest-gpu:
	uv run coverage run -m pytest --maxfail=10

coverage-report: unittest
	uv run coverage report
	uv run coverage html

test: check unittest
