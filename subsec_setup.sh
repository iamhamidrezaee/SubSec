#!/bin/bash

conda create -n subsec python=3.11
conda activate subsec
pip install -r requirements.txt
# Ensure torchvision is installed via pip (not system package)
pip install --force-reinstall torchvision