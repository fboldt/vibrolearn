from tests.sehri_et_al_cwru import print_papers_sehri_meta_data
from experiment.sehri_et_al import run_sehri_et_al_experiment, get_wavelet_random_forest


if __name__ == "__main__":
    print_papers_sehri_meta_data()
    model = get_wavelet_random_forest()
    run_sehri_et_al_experiment(model)
