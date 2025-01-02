from scripts.dbua import dbua
import torch
from utilities.data import SAMPLE, LOSS

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dbua(SAMPLE, LOSS, device)