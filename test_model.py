import os
from pipes import Template
import sys
from telnetlib import Telnet
os.chdir(sys.path[0])
os.environ["CUDA_VISIBLE_DEVICES"]='1'
import torch

from model.model import TEMPLATE
from model.utils import load_and_cache_withlabel,PrintModelInfo,CaculateAcc, load_checkpoint
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH="./output/output_model/.ckpt"


def main():
    model = Template().to(DEVICE)
    ckpt = load_checkpoint(MODEL_PATH)
    model.load_state_dict(ckpt['model'])
    PrintModelInfo(model)
    print("model load weight done.")
    model.eval()
    with torch.no_grad():
        output = model()
    
if __name__=="__main__":
    main()