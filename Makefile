# clean and reinstall poetry environment
# ensure correct pyenv version is installed
# ensure correct poetry version is installed
# make sure pyenv links to poetry project
# python version saved in .python-version file
python_version = $(shell cat .python-version)


.PHONY: clean
clean:
	@rm -rf .pytest_cache
	@rm -rf .mypy_cache

.PHONY: reset
reset: clean
	@echo "Cleaning up..."
	@rm -rf .venv
	@poetry env use /Users/$(USER)/.pyenv/versions/${python_version}/bin/python
	@poetry install
	@poetry run python -m pip install --upgrade pip
	@poetry run python -m pip install tensorflow-macos
	@poetry run python -m pip install tensorflow-metal
	@echo "Done."
