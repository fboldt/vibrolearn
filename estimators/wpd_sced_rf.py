from itertools import product

from sklearn.ensemble import RandomForestClassifier

from estimators.pipeline import Pipeline
from feature.extraction import WaveletFeatures
from feature.feature_selector import StableDomainFeatureSelector


class WPD_SCED_RF:

    K_VALUES = (8,)
    ALPHA_VALUES = (0.7,)

    @property
    def name(self):
        return "wpd_sced_rf"

    def configurations(self):
        """
        Retorna as configurações que devem ser executadas.
        """

        for k, alpha in product(
            self.K_VALUES,
            self.ALPHA_VALUES
        ):
            yield {
                "k": k,
                "alpha": alpha
            }

    def build(self, configuration):

        k = configuration["k"]
        alpha = configuration["alpha"]

        steps = [
            (
                "feature_extraction",
                WaveletFeatures()
            ),
            (
                "feature_selection",
                StableDomainFeatureSelector(
                    k=k,
                    alpha=alpha
                )
            ),
            (
                "classifier",
                RandomForestClassifier()
            )
        ]

        return Pipeline(steps)

    def metadata(self, configuration):
        return {
            "method": self.name,
            "k": configuration["k"],
            "alpha": configuration["alpha"]
        }