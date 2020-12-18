#!/bin/bash

mkdir ./cassava_venv

virtualenv --python=/usr/bin/python3.8 ./cassava_venv/

source ./cassava_venv/bin/activate
pip install -r requirements-dev.txt
