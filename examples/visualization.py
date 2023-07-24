import optuna

import optuna_distributed


def objective(trial):
    x = trial.suggest_float("x", -100, 100)
    y = trial.suggest_categorical("y", [-1, 0, 1])
    return x**2 + y


sampler = optuna.samplers.TPESampler(seed=10)
distributed_study = optuna_distributed.from_study(optuna.create_study(sampler=sampler))
distributed_study.optimize(objective, n_trials=30)

# Any plotting function from optuna.visualization module can be used with Optuna-distributed
# thanks to .into_study() convenience method.
optuna.visualization.plot_contour(distributed_study.into_study(), params=["x", "y"]).show()
