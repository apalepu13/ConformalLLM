#################################################################################
# PROJECT STRUCTURE                                                             #
#################################################################################

PROJECT_DIR := $(addprefix $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST)))), /)
PYTHON_INTERPRETER = python3

# Set up the directory structure
SRC_DIR = $(addprefix $(PROJECT_DIR), src/)
RUN_DIR := $(addprefix $(PROJECT_DIR), run/)
LOG_DIR := $(addprefix $(RUN_DIR), logs/)
RESULTS_DIR := $(addprefix $(DATA_DIR), results/)
SAVED_MODELS_DIR := $(addprefix $(PROJECT_DIR), models/)
CONFIG_DIR := $(addprefix $(PROJECT_DIR), configs/)
eval_config := $(addprefix $(CONFIG_DIR), config.json)

#################################################################################
# EXPERIMENTAL PARAMETERS                                                       #
#################################################################################
# Set random seed for experiments (ensures reproducibility)
RANDOM_SEED = 13

all: evaluate

#################################################################################
# Setup requirements                                                            #
#################################################################################
requirements: python_requirements.txt
	pip install -r python_requirements.txt

#################################################################################
# Evaluations				                                                        #
#################################################################################

evaluate: self_consistency

self_consistency:
	$(PYTHON_INTERPRETER) $(addprefix $(SRC_DIR), evaluation/self_consistency_test.py) $(eval_config)