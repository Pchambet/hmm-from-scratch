.PHONY: setup data run report test lint all

setup:  ## install the locked environment
	uv sync --locked

data:  ## check the committed data and download FSDD (16 MB, once)
	uv run hmm-markov data

run:  ## both experiments, ~15 min on a laptop (results/*.json)
	uv run hmm-markov rain
	uv run hmm-markov words

report:  ## figures in docs/figures and the page in site/index.html
	uv run hmm-markov report

test:
	uv run pytest -q

lint:
	uv run ruff check .
	uv run ruff format --check .

all: setup data run report
