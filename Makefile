# PRISM Phase 1 pipeline:  make setup data backtest assets
.PHONY: setup data test smoke backtest assets notebooks all clean-cache

setup:          ## create the virtual environment
	uv sync

data:           ## download M5 into data/raw
	./scripts/download_data.sh

test:           ## unit tests (synthetic data, no download needed)
	uv run pytest -q

smoke:          ## 60 series, 1 origin, about 1 minute
	uv run python scripts/run_backtest.py --smoke

backtest:       ## full rolling-origin backtest (1,500 series x 5 origins, about 20 min)
	uv run python scripts/run_backtest.py

assets:         ## report tables, figures and headline numbers
	uv run python scripts/make_report_assets.py

notebooks:      ## execute all notebooks in place
	for nb in notebooks/0*.ipynb; do uv run jupyter nbconvert --to notebook --execute --inplace $$nb; done

all: data test backtest assets notebooks

clean-cache:    ## force all models to refit
	rm -rf results/cache
