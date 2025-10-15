from experiment.cwru_rauber_loca_et_al import run
from utils.time import measure_time
from estimators import rfwfe


if __name__ == "__main__":
    print("Running Random Forest with WFE...")
    result, elapsed = measure_time(run, rfwfe.model, verbose=True)
    print(f"Execution time: {elapsed:.2f} seconds")

