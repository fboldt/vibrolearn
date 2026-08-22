from dataset.loader import vanilla, augmented

from preprocessing.data_adapter import (
    DataAdapter,
    PreparedData
)


class SignalAdapter(DataAdapter):

    def __init__(
        self,
        segment_length,
        use_domains=False,
        augment_training=False
    ):
        self.segment_length = segment_length
        self.use_domains = use_domains
        self.augment_training = augment_training

    def prepare(
        self,
        registers,
        experimental_setup,
        training=False
    ):
        # Cria uma cópia para que o método possa definir
        # seu próprio tamanho de segmento.
        setup = experimental_setup.copy()

        setup["segment_length"] = (
            self.segment_length
        )

        loader = vanilla

        if (
            training
            and self.augment_training
        ):
            loader = augmented

        loaded_data = loader(
            registers,
            setup,
            get_domains=self.use_domains
        )

        if self.use_domains:
            X, y, domains = loaded_data

            return PreparedData(
                X=X,
                y=y,
                domains=domains
            )

        X, y = loaded_data

        return PreparedData(
            X=X,
            y=y
        )