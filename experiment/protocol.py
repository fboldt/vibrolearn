import json

from pathlib import Path


class ExperimentalProtocol:

    def __init__(
        self,
        path
    ):
        self.path = Path(path)

        self.config = (
            self._load_json(
                self.path
            )
        )

        dataset_path = (
            self.config.get(
                "dataset"
            )
        )

        if dataset_path is None:
            self.dataset_config = {}
        else:
            self.dataset_config = (
                self._load_json(
                    Path(dataset_path)
                )
            )

    @staticmethod
    def _load_json(path):

        with open(
            path,
            "r",
            encoding="utf-8"
        ) as file:
            return json.load(file)

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

        setup = {
            **self.dataset_config,
            **self.config
        }

        setup.pop(
            "dataset",
            None
        )

        return setup