from scripts.dbua import dbua
from utilities.data import SAMPLE, LOSS, CTRUE

if __name__ == "__main__":
    # dbua(SAMPLE, LOSS)
    run_list = [
        "1420", "1465", "1480", "1510", "1540", "1555", "1570",
        "inclusion", "inclusion_layer", "four_layer", "two_layer", 
        "checker2", "checker8"
    ]
    run_list = ["inclusion_layer"]
    # Run all examples
    for sample in run_list:
        print(sample)
        dbua(sample, LOSS)