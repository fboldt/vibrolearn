import json
from pathlib import Path


class ExperimentalProtocol:

    def __init__(self, path):
        self.path = Path(path)

        with open(
            self.path,
            "r",
            encoding="utf-8"
        ) as file:
            self.config = json.load(file)

    @property
    def name(self):
        return self.config.get(
            "experiment_name",
            self.path.stem
        )

    @property
    def repetitions(self):
        return self.config.get(
            "repetitions",
            1
        )

    @property
    def output_dir(self):
        return Path(
            self.config.get(
                "output_dir",
                "results"
            )
        )

    def get_setup(self):
        return self.config