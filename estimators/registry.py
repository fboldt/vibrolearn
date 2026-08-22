from estimators.wpd_sced_rf import WPD_SCED_RF
from estimators.dinov2 import DINOv2Method
from estimators.cnn_lstm import CNNLSTMMethod


METHODS = {
    "wpd_sced_rf": WPD_SCED_RF,
    "dinov2": DINOv2Method,
    "cnn_lstm": CNNLSTMMethod,
}


def get_method(name):
    try:
        method_class = METHODS[name]
    except KeyError:
        available = ", ".join(METHODS.keys())

        raise ValueError(
            f"Unknown method '{name}'. "
            f"Available methods: {available}"
        )

    return method_class()