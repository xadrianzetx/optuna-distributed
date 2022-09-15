"""
This example adds Optuna-distributed semantics on top of
https://github.com/optuna/optuna-examples/blob/main/simple_pruning.py

Optuna example that demonstrates a pruner.
In this example, we optimize a classifier configuration using scikit-learn. Note that, to enable
the pruning feature, the following 2 methods are invoked after each step of the iterative training.
(1) :func:`optuna.trial.Trial.report`
(2) :func:`optuna.trial.Trial.should_prune`
You can run this example as follows:
    $ python simple_prunning.py
"""

from dask.distributed import Client
import optuna
from optuna.trial import TrialState
import sklearn.datasets
import sklearn.linear_model
import sklearn.model_selection

import optuna_distributed


def objective(trial):
    iris = sklearn.datasets.load_iris()
    classes = list(set(iris.target))
    train_x, valid_x, train_y, valid_y = sklearn.model_selection.train_test_split(
        iris.data, iris.target, test_size=0.25
    )

    alpha = trial.suggest_float("alpha", 1e-5, 1e-1, log=True)
    clf = sklearn.linear_model.SGDClassifier(alpha=alpha)

    for step in range(100):
        clf.partial_fit(train_x, train_y, classes=classes)

        # Report intermediate objective value.
        intermediate_value = clf.score(valid_x, valid_y)
        trial.report(intermediate_value, step)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.TrialPruned()

    return clf.score(valid_x, valid_y)


if __name__ == "__main__":
    # Optuna-distributed just wraps standard Optuna study. The resulting object behaves
    # just like regular study, but optimization process is asynchronous.
    study = optuna_distributed.from_study(
        optuna.create_study(direction="maximize"), client=Client()
    )
    study.optimize(objective, n_trials=100)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
