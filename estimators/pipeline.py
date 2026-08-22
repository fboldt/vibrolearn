from sklearn.pipeline import (
    Pipeline as SklearnPipeline
)

from timed_decorator.simple_timed import (
    timed
)


@timed(
    return_time=True,
    use_seconds=True
)
def timed_prepare(
    adapter,
    registers,
    experimental_setup,
    training
):
    return adapter.prepare(
        registers,
        experimental_setup,
        training=training
    )


@timed(
    return_time=True,
    use_seconds=True
)
def timed_fit(
    pipe,
    X,
    y,
    domains=None
):
    if domains is not None:
        return pipe.fit(
            X,
            y,
            feature_selection__domains=domains
        )

    return pipe.fit(
        X,
        y
    )


@timed(
    return_time=True,
    use_seconds=True
)
def timed_predict(
    pipe,
    X
):
    return pipe.predict(X)


class Pipeline:

    def __init__(
        self,
        steps,
        data_adapter
    ):
        self.pipe = (
            SklearnPipeline(
                steps
            )
        )

        self.data_adapter = (
            data_adapter
        )

        self.scores = {}

    # ========================================================
    # TRAIN
    # ========================================================

    def train(
        self,
        registers,
        experimental_setup
    ):
        self.experimental_setup = (
            experimental_setup
        )

        data, load_time = (
            timed_prepare(
                self.data_adapter,
                registers,
                experimental_setup,
                training=True
            )
        )

        self.scores[
            "load_data_time"
        ] = load_time

        _, training_time = timed_fit(
            self.pipe,
            data.X,
            data.y,
            domains=data.domains
        )

        self.scores[
            "training_time"
        ] = training_time

        return self

    # ========================================================
    # EVALUATE
    # ========================================================

    def evaluate(
        self,
        registers,
        metrics
    ):
        data, _ = timed_prepare(
            self.data_adapter,
            registers,
            self.experimental_setup,
            training=False
        )

        y_pred, prediction_time = (
            timed_predict(
                self.pipe,
                data.X
            )
        )

        self.scores[
            "prediction_time"
        ] = prediction_time

        for metric in metrics:

            self.scores[
                metric.__name__
            ] = metric(
                data.y,
                y_pred
            )

        return self.scores