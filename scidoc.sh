#!/bin/bash
# SciDoc Runner

# Suppress warnings
export TF_CPP_MIN_LOG_LEVEL=3
export CUDA_VISIBLE_DEVICES=""
export TF_ENABLE_ONEDNN_OPTS=0
export TF_ENABLE_DEPRECATION_WARNINGS=0
export PYTHONWARNINGS="ignore"

# Run SciDoc
conda run -n scidoc scidoc "$@" 2>/dev/null
exit $?
