# Disable all built in rules
.SUFFIXES:
MAKEFLAGS += --no-builtin-rules

# Allow bash syntax
SHELL := /bin/bash

test ::
	uv run pytest tests


venv ::
	@if ! command -v uv &> /dev/null; then \
		echo "Installing uv..."; \
		curl -LsSf https://astral.sh/uv/install.sh | sh > /dev/null 2>&1 || { echo "Error: Failed to install uv"; exit 1; }; \
		echo "uv installed successfully"; \
	fi
	@if [ ! -d .venv ]; then \
		echo "Creating virtual environment..."; \
		uv venv > /dev/null 2>&1 || { echo "Error: Failed to create venv"; exit 1; }; \
		echo "Virtual environment created"; \
	fi

compile :: venv
	@./tools/check_compile.sh
	@uv pip uninstall . > /dev/null 2>&1 || true
	@echo "Installing package..."
	@rm -rf build dist src/*.egg-info
	@rm -f ./src/ember/fortran.*.so
	@uv pip install -e . --force-reinstall > /dev/null  || { echo "Error: Package installation failed"; exit 1; }
	@echo "Package installed successfully"

ci ::
	act push

docs ::
	uv run sphinx-build -W -b html docs docs/_build/html

docs-full ::
	uv run sphinx-build -W -b html -D sphinx_gallery_conf.filename_pattern='.*\.py' docs docs/_build/html

docs-clean ::
	rm -rf docs/_build
