import copy
import itertools

from src import config_utils
from train import log_hyperparameters, main  # reuse your functions

BASE_CFG_PATH = "config.yaml"


def override(cfg, **kv):
    out = copy.deepcopy(cfg)
    for k, v in kv.items():
        out[k] = v
    return out


def make_run_id(cfg, extra=""):
    return f"{cfg['run_id']}_{extra}" if extra else cfg["run_id"]


if __name__ == "__main__":
    base = config_utils.load_config(BASE_CFG_PATH)

    sweep = {"STARTING_LR": [1e-2, 1e-3], "ALPHA": [0.1, 0.01]}

    keys = list(sweep.keys())
    for values in itertools.product(*[sweep[k] for k in keys]):
        params = dict(zip(keys, values))
        cfg = override(base, **params)

        tag = "_".join(f"{k}{v}" for k, v in params.items())
        cfg["run_id"] = make_run_id(base, tag)

        log_hyperparameters(cfg)
        main(cfg)
