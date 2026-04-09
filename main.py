from experiment.rauber_loca_et_al import run_rauber_loca_et_al_experiment
from experiment.sehri_et_al import run_sehri_et_al_papers, run_sehri_et_al_papers_inspired_experiment, run_sehri_et_al_proposed_experiment
from estimators.wavelet_random_forest import WaveletRandomForest
# from experiment.sehri_et_al_aug import run_papers_experiment_augmented

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run vibrolearn with the following options:")
    parser.add_argument("-d", "--debug", action="store_true", help="Run in debug mode")
    parser.add_argument("-rf", "--random_forest", action="store_true", help="Use the wavelet random forest model")
    parser.add_argument("-sp", "--sehri_et_al_papers", action="store_true", help="Run the experiment")
    # parser.add_argument("-spa", "--sehri_et_al_papers_augmented", action="store_true", help="Run the augmented experiment")
    parser.add_argument("-spi", "--sehri_et_al_papers_inspired", action="store_true", help="Run the experiment inspired by the paper's cross validation folds")
    parser.add_argument("-spp", "--sehri_et_al_proposed", action="store_true", help="Run the experiment with the proposed cross validation folds")
    
    args = parser.parse_args()
    if not any(vars(args).values()):
        parser.print_help()
    
    model = None
    if args.debug:
        print("Running in debug mode")
        model = WaveletRandomForest(42)
        run_rauber_loca_et_al_experiment(model)
    if args.random_forest:
        model = WaveletRandomForest(42)
    if args.sehri_et_al_papers:
        run_sehri_et_al_papers(model)
    # if args.sehri_et_al_papers_augmented:
    #     run_papers_experiment_augmented(model, mixes_per_pair=5, seed=0, augment=True)
    if args.sehri_et_al_papers_inspired:
        run_sehri_et_al_papers_inspired_experiment(model)
    if args.sehri_et_al_proposed:
        run_sehri_et_al_proposed_experiment(model)