import argparse

from train import train


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    config = {
    'batch_size': 16,
    'dropout_rate': 0.8,
    'factor': 4,
    'kernel_size': 63,
    'lr': 0.001,
    'n_epochs': 200,
    'n_res_blks': 3,
    'out_channels': 16,
    }

    train(config, 0)

