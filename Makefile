SHELL := /bin/bash
VIRTUALENV_DIR ?= venv
PYTHON_VERSION = python3.6


.PHONY: virtualenv
virtualenv:
	@echo
	@echo "==================== virtualenv ===================="
	@echo
	test -f $(VIRTUALENV_DIR)/bin/activate || python3.6 -m venv $(VIRTUALENV_DIR)

.PHONY: requirements
requirements: virtualenv
	@echo
	@echo "==================== requirements ===================="
	@echo
	# Install requirements
	. $(VIRTUALENV_DIR)/bin/activate; $(VIRTUALENV_DIR)/bin/pip install -r requirements.txt

.PHONY: lint-requirements
lint-requirements: requirements
	@echo
	@echo "==================== lint requirements ===================="
	@echo
	# Install requirements
	. $(VIRTUALENV_DIR)/bin/activate; $(VIRTUALENV_DIR)/bin/pip install pylint flake8

.PHONY: lint
lint: .flake8 .pylint

.PHONY: .pylint
.pylint:
	@echo
	@echo "================== pylint ===================="
	@echo
	. $(VIRTUALENV_DIR)/bin/activate; PYTHONPATH=. pylint -E *.py

.PHONY: .flake8
.flake8:
	@echo
	@echo "==================== flake ===================="
	@echo
	. $(VIRTUALENV_DIR)/bin/activate; flake8 *.py
