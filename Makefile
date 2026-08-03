# Common entry points. Run `make help` for the list.
# Everything uses PYTHONPATH=. so the top-level `src` package imports cleanly.
.PHONY: help install test lint figures clean

export PYTHONPATH := .

help:                ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN{FS=":.*?## "}{printf "  \033[1m%-12s\033[0m %s\n", $$1, $$2}'

install:             ## Install pinned dependencies
	pip install -r requirements.txt

test:                ## Run the test suite (metrics, economics, leakage)
	pytest

lint:                ## Static checks (unused imports, style)
	python -m pyflakes src tests

figures:             ## Regenerate the figures in docs/img
	python -m src.plots

clean:               ## Remove caches and generated artifacts (keeps committed results)
	rm -rf outputs/cache/* outputs/*.log **/__pycache__ .pytest_cache
