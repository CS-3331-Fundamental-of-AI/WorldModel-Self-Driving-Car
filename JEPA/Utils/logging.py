# utils/logging.py

def log_metrics(experiment, metrics: dict, prefix: str, step: int):
    """
    Log a dict of metrics to Comet under a common prefix.

    Example:
        log_metrics(exp, {"total": 1.2}, prefix="loss/jepa1", step=10)

    → logs: loss/jepa1/total
    """
    for k, v in metrics.items():
        if v is None:
            continue
        experiment.log_metric(
            f"{prefix}/{k}",
            float(v),
            step=step,
        )
