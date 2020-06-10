import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", "-o", default="output.csv")
    parser.add_argument("--model_name", "-m")
    parser.add_argument("--dim", default=128, type=int)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--batch_size", default=1024, type=int)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--bce", action="store_true")
    args = parser.parse_args()

    return args
