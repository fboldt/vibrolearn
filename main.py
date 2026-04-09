from experiment.rauber_loca_et_al import run_rauber_loca_et_al_experiment
from experiment.sehri_et_al import run_sehri_et_al_papers, run_sehri_et_al_papers_inspired_experiment, run_sehri_et_al_proposed_experiment
from estimators.wavelet_random_forest import WaveletRandomForest

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run vibrolearn with the following options:")
    parser.add_argument("-d", "--debug", action="store_true", help="Run in debug mode")
    parser.add_argument("-rfc", "--random_forest", action="store_true", help="Use the wavelet random forest model")
    parser.add_argument("-rac", "--random_forest_augmented", action="store_true", help="Use the wavelet random forest model with augmented data")
    parser.add_argument("-re", "--rauber_loca_et_al_experiment", action="store_true", help="Run the experiment from Rauber et al.")
    parser.add_argument("-se", "--sehri_et_al_papers_experiment", action="store_true", help="Run the experiment from Sehri et al.")
    parser.add_argument("-sie", "--sehri_et_al_papers_inspired_experiment", action="store_true", help="Run the experiment inspired by the paper's cross validation folds")
    parser.add_argument("-spe", "--sehri_et_al_proposed_experiment", action="store_true", help="Run the experiment with the proposed cross validation folds")

    args = parser.parse_args()
    if not any(vars(args).values()):
        parser.print_help()
    
    model = None
    if args.debug:
        print("Running in debug mode")
    if args.random_forest:
        model = WaveletRandomForest(42)
    if args.random_forest_augmented:
        model = WaveletRandomForest(42)
    if args.rauber_loca_et_al_experiment:
        run_rauber_loca_et_al_experiment(model)
    if args.sehri_et_al_papers_experiment:
        run_sehri_et_al_papers(model)
    if args.sehri_et_al_papers_inspired_experiment:
        run_sehri_et_al_papers_inspired_experiment(model)
    if args.sehri_et_al_proposed_experiment:
        run_sehri_et_al_proposed_experiment(model)
