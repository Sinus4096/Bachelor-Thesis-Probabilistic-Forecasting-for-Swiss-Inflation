import sys
import argparse
import yaml
#run experiments
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