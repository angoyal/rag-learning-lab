"""MLflow logging helpers for experiments and test runs."""


def log_experiment_run(experiment_name: str, params: dict, metrics: dict) -> None:
    raise NotImplementedError


def log_test_run(test_name: str, passed: bool, duration_s: float) -> None:
    raise NotImplementedError
