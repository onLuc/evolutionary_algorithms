tuning.py saves the best configuration to best_params.json
GA.py tries to load the configuration from best_params.json, but falls back to a default configuration if the file is not found so make sure the .json file is in the same working directory as the script.
ES.py is a standalone script.

requirements.txt list the required packages and libraries for executing the scripts.