# clean and reinstall poetry environment
# ensure correct pyenv version is installed
# ensure correct poetry version is installed
# make sure pyenv links to poetry project
.PHONY: clean
clean:
	@rm -rf .pytest_cache
	@rm -rf .mypy_cache

.PHONY: reset
reset: clean
	@echo "Cleaning up..."
	@rm -rf .venv
	@poetry env use /Users/$(USER)/.pyenv/versions/3.10.13/bin/python
	@poetry install