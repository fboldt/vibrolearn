from experiment.cwru_rauber_loca_et_al import run
from utils.time import measure_time

if __name__ == "__main__":
    result, elapsed = measure_time(run)
    print(f"Execution time: {elapsed:.2f} seconds")
