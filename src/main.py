import argparse

from train import train


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    config = {
        "lr": 1e-3,
        "kernel_size": 31,
        "n_res_blks": 4,
        "dropout_rate": 0.5,
        "batch_size": 64,
        "out_channels": 1,
        "factor": 8,
    }

    train(config, 0)

