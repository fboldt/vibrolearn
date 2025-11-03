from estimators import rfwfe, adaboost, randomforest
from experiment.cwru_rauber_loca_et_al import run
from pprint import pprint


if __name__ == "__main__":
    print("Running Random Forest with Raw Features...")
    result  = run(randomforest.model, verbose=True)

    print("\nRunning Random Forest with Heterogeneous Features...")
    result  = run(rfwfe.model, verbose=True)

    print("\nRunning AdaBoost with Heterogeneous Features...")
    result = run(adaboost.model, verbose=True)


