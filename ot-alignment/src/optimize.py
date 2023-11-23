from ax.service.managed_loop import optimize

from unbalanced_minibatch import train_eval_fn


if __name__ == "__main__":
    bp, val, exp, mod = optimize(
        parameters=[
            {
                "name": "x1",
                 "type": "choice",
                 "values": [17, 19, 23, 29, 31],
                 "value_type": "int"},
            {
                "name": "x2",
                "type": "range",
                "bounds": [1e-5, 1e-1],
                "value_type": "float",
                "log_scale": False
            },
            {
                "name": "x3",
                "type": "range",
                "bounds": [0, 1],
                "value_type": "float",
                "log_scale": False
            },
            {
                "name": "x4",
                "type": "fixed",
                "value_type": "int",
                "value": 2048
            },
            {
                "name": "x5",
                "type": "fixed",
                "value_type": "int",
                "value": 5000
            }
        ],
        experiment_name="trial_alpha",
        objective_name="unbalanced_minibatch",
        evaluation_function=train_eval_fn,
        minimize=True,
        total_trials=100
    )
    print(bp)
    print(val)
