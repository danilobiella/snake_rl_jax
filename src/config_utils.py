import yaml

from src.types import ActorCriticConfig, TrainingConfig
from src.game import game


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def create_actor_critic_config(config: dict) -> ActorCriticConfig:
    return ActorCriticConfig(
        n_agents=config["N_AGENTS"],
        gamma=config["GAMMA"],
        beta=config["BETA"],
        t_max=config["T_MAX"],
    )


def create_training_config(config: dict) -> TrainingConfig:
    filename = (
        f"results/{game.GRID_SIZE}x{game.GRID_SIZE}/{config['run_id']}.csv"
    )
    return TrainingConfig(
        run_id=config["run_id"],
        num_screens_to_play=config["NUM_SCREENS_TO_PLAY"],
        loops_before_metrics=config["LOOPS_BEFORE_METRICS"],
        num_snake_length_evaluations=config["NUM_SNAKE_LENGHT_EVALUATIONS"],
        metrics_filename=filename,
        num_games_checkpoint=config["NUM_GAMES_CHECKPOINT"],
    )
