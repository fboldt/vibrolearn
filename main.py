import argparse

from experiment.runner import run


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run VibroLearn experiments."
    )

    parser.add_argument(
        "-m",
        "--method",
        required=True,
        help="Method used in the experiment."
    )

    parser.add_argument(
        "-e",
        "--experimental_setup",
        required=True,
        help="Experimental protocol configuration file."
    )

    return parser.parse_args()


def main():
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()