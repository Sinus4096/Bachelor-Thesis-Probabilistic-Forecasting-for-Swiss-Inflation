import os
import sys
from pathlib import Path
#run experiments
from Code.Scripts.Models.qrf_model import run_experiment
# 1. FORCE the project root into Python's search path IMMEDIATELY
project_root = str(Path(__file__).resolve().parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 2. NOW you can import your custom code safely
import yaml
import argparse
from Code.Scripts.Models.qrf_model import run_experiment

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

if __name__=="__main__":
    parser =argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the config file")
    args =parser.parse_args()
    
    #load and run
    config_data= load_config(args.config)
    run_experiment(config_data)