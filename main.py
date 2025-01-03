from scripts.dbua import dbua
from utilities.data import SAMPLE, LOSS, CTRUE

if __name__ == "__main__":
    # dbua(SAMPLE, LOSS)

    # Run all examples
    for sample in CTRUE.keys():
        print(sample)
        dbua(sample, LOSS)