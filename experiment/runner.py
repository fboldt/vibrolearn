from datetime import datetime

from estimators.registry import get_method
from experiment.protocol import ExperimentalProtocol
from experiment.assesment import (
    run_experiment,
    save_scores
)


def run(args):

    # --------------------------------------------------------
    # Obtém método e protocolo
    # --------------------------------------------------------

    method = get_method(
        args.method
    )

    protocol = ExperimentalProtocol(
        args.experimental_setup
    )

    # --------------------------------------------------------
    # Informações da execução
    # --------------------------------------------------------

    print(
        f"\nMethod: {method.name}"
    )

    print(
        f"Protocol: {protocol.name}"
    )

    print(
        f"Repetitions: {protocol.repetitions}"
    )

    print(
        f"Output directory: "
        f"{protocol.output_dir}\n"
    )

    # --------------------------------------------------------
    # Configurações fornecidas pelo próprio método
    # --------------------------------------------------------

    for configuration in method.configurations():

        execute_configuration(
            method=method,
            protocol=protocol,
            configuration=configuration
        )


def execute_configuration(
    method,
    protocol,
    configuration
):

    for repetition in range(
        1,
        protocol.repetitions + 1
    ):

        print(
            f"\n{'=' * 60}"
        )

        print(
            f"Repetition "
            f"{repetition}/"
            f"{protocol.repetitions}"
        )

        if configuration:
            print(
                f"Configuration: "
                f"{configuration}"
            )

        print(
            f"{'=' * 60}"
        )

        # ----------------------------------------------------
        # O método constrói seu próprio pipeline
        # ----------------------------------------------------

        pipeline = method.build(
            configuration
        )

        # ----------------------------------------------------
        # Executa
        # ----------------------------------------------------

        scores = run_experiment(
            pipeline,
            protocol.get_setup()
        )

        # ----------------------------------------------------
        # Resultado
        # ----------------------------------------------------

        save_result(
            method=method,
            protocol=protocol,
            configuration=configuration,
            repetition=repetition,
            scores=scores
        )


def save_result(
    method,
    protocol,
    configuration,
    repetition,
    scores
):

    timestamp = datetime.now().strftime(
        "%Y-%m-%d_%H-%M-%S"
    )

    results = {
        "experiment_name": protocol.name,
        "method": method.name,
        "repetition": repetition,
        "scores": scores,
        "end_time": timestamp
    }

    # --------------------------------------------------------
    # Cada método informa seus próprios metadados
    # --------------------------------------------------------

    results.update(
        method.metadata(
            configuration
        )
    )

    # --------------------------------------------------------
    # Saída
    # --------------------------------------------------------

    output_file = (
        protocol.output_dir
        /
        f"run-{repetition:02d}_{timestamp}.json"
    )

    save_scores(
        results,
        str(output_file)
    )