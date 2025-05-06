#!/bin/bash

# Prompt for input values
read -p "Prompt to Generate Music:      " user_prompt
read -p "Model Name (mistral | gpt2):   " model_name
read -p "Enter temperature (e.g., 0.6): " temperature

# Run the Python script with user input
python -m src.deploy.handler \
  --test_input "{\"input\": {\"prompt\": \"$user_prompt\", \"model\": \"$model_name\", \"temperature\": $temperature, \"max_tokens\": 1024}}"