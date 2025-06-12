# Persona-Twin LLM

This repository contains code for training and evaluating large language models (LLMs) to simulate psychological patient responses based on demographic, behavioral, and psychological data.

## Overview

The project explores how well LLMs can simulate patient responses in psychological contexts by conditioning them on different types of patient data:

- Demographic data (age, gender, education, etc.)
- Behavioral data (health behaviors, exercise habits, etc.) 
- Psychological data (personality traits, mental health indicators, etc.)

## Repository Structure

```
📂 Data/                      # Contains survey data and response datasets
📂 json_datasets/            # Generated JSON datasets for model training
📂 results/                  # Output results from experiments
📂 plots/                    # Generated visualization plots
📜 Individual_LLama3.ipynb  # Main notebook for Llama 3 experiments
📜 Individual_4o.ipynb  # Main notebook for GPT-4o experiments
📜 Fairness_Plots.ipynb     # Notebook for generating fairness analysis plots
📜 requirements.txt         # Python package dependencies
📜 LICENSE                 # Project license information
📜 README.md               # Project documentation
```

## Features

- Data preprocessing and formatting for LLM training
- Generation of patient responses using Llama 3 70B model
- Multiple experimental conditions:
  - All data conditioning
  - Holdout experiments (leaving out one response type)
  - No system prompt baseline
- Evaluation of response quality and consistency

## Setup & Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Install Poetry for dependency management:
```bash
# Install pipx
python -m pip install --user pipx
python -m pipx ensurepath

# Install Poetry using pipx
pipx install poetry
```

3. Install project dependencies using Poetry:
```bash
poetry install
```

## Usage

The main scripts are:

- `json_files_generation.py` - Generates training data in JSONL format
- `Individual_LLama3.ipynb` - Notebook for running experiments with Llama 3
- `Individual_4o.ipynb`  - Notebook for running experiments with GPT-4o

### API Keys Required

To run the experiments, you'll need API keys for:
- Replicate (for Llama 3 access)
- OpenAI (optional, for GPT model comparisons)

Set up your API keys as environment variables:
```bash
export REPLICATE_API_TOKEN="your_replicate_api_token_here"
export OPENAI_API_KEY="your_openai_api_key_here"
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

```
# Contributors

This project is maintained by the Human-centered Analytics Lab (HAL).
