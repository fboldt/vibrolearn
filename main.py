from tests.sehri_et_al_cwru import print_papers_sehri_meta_data
from experiment.sehri_et_al import get_wavelet_random_forest, run_sehri_et_al_papers, run_sehri_et_al_papers_inspired_experiment, run_sehri_et_al_proposed_experiment

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run vibrolearn with the following options:")
    parser.add_argument("--metadata", action="store_true", help="Print papers metadata")
    parser.add_argument("-rf", "--random_forest", action="store_true", help="Use the wavelet random forest model")
    parser.add_argument("-sp", "--sehri_et_al_papers", action="store_true", help="Run the experiment")
    parser.add_argument("-spi", "--sehri_et_al_papers_inspired", action="store_true", help="Run the experiment inspired by the paper's cross validation folds")
    parser.add_argument("-spp", "--sehri_et_al_proposed", action="store_true", help="Run the experiment with the proposed cross validation folds")
    
    args = parser.parse_args()
    if not any(vars(args).values()):
        parser.print_help()
    
    model = None
    if args.metadata:
        print_papers_sehri_meta_data()
    if args.random_forest:
        model = get_wavelet_random_forest()
    if args.sehri_et_al_papers:
        run_sehri_et_al_papers(model)
    if args.sehri_et_al_papers_inspired:
        run_sehri_et_al_papers_inspired_experiment(model)
    if args.sehri_et_al_proposed:
        run_sehri_et_al_proposed_experiment(model)