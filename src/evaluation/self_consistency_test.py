
import os
import argparse
import json
import torch
import sys
sys.path.insert(1, os.path.join(os.path.dirname(__file__), ""))
import eval_utils
sys.path.insert(1, os.path.join(os.path.dirname(__file__), "../datasets"))
print(sys.path)
import MedQA
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main():
    #Setup config and wandb
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str)
    args = parser.parse_args()
    with open(args.config_path, "r") as f:
        config = json.load(f)
    torch.manual_seed(config['random_seed'])
    tokenizer, model = eval_utils.getModel(config)

    MedQA_dat = MedQA.get_MedQA_loader(config, subset="all")
    eval_utils.self_consistency(MedQA_dat, tokenizer, model, config, name="MedQA")

if __name__ == "__main__":
    main()


