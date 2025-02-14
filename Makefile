#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = uncertainty_ellipsoid
PYTHON_VERSION = 3.10
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python Dependencies
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt
	



## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using flake8 and black (use `make format` to do formatting)
.PHONY: lint
lint:
	flake8 uncertainty_ellipsoid
	isort --check --diff --profile black uncertainty_ellipsoid
	black --check --config pyproject.toml uncertainty_ellipsoid

## Format source code with black
.PHONY: format
format:
	black --config pyproject.toml uncertainty_ellipsoid
	isort --profile black uncertainty_ellipsoid




## Set up python interpreter environment
.PHONY: create_environment
create_environment:
	
	conda create --name $(PROJECT_NAME) python=$(PYTHON_VERSION) -y
	
	@echo ">>> conda env created. Activate with:\nconda activate $(PROJECT_NAME)"
	



#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Make Dataset
.PHONY: data
data: requirements
	$(PYTHON_INTERPRETER) uncertainty_ellipsoid/dataset.py

## Generate Features (split train/test)
.PHONY: features
features: 
	$(PYTHON_INTERPRETER) uncertainty_ellipsoid/features.py

## Make Train
.PHONY: train
train: 
	$(PYTHON_INTERPRETER) uncertainty_ellipsoid/modeling/train.py

## Make tensorboard
.PHONY: tensorboard
tensorboard: 
	tensorboard --logdir ./runs


## Make Predict
.PHONY: predict
predict: 
	$(PYTHON_INTERPRETER) uncertainty_ellipsoid/modeling/predict.py


## Make Api
.PHONY: api
api: 
	uvicorn uncertainty_ellipsoid.api:app --reload
	
#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
