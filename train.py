import argparse
import json
import pathlib

import haiku as hk
import jax
import optax  # type: ignore

from src import actor_critic, config_utils, func_approx, models, training
from src.game import game, game_with_history


def initialize_model(model_name: str):
    model = models.get_model(model_name, game.GRID_SIZE)

    print("Using Model:")
    print(
        hk.experimental.tabulate(
            model, columns=["module", "input", "output", "params_size", "params_bytes"]
        )(
            func_approx.into_tensor_representation(
                game_with_history.random_state(jax.random.PRNGKey(6))
            )
        )
    )
    return model


def setup_optimizer(starting_lr: float, decay_steps: int, alpha: float, clip: float):
    print("-" * 30)
    print("Setting up optimizer")
    print(f"Optimizer starting learning rate: {starting_lr:.2e}")
    print(f"Optimizer final learning rate: {starting_lr * alpha:.2e}")
    print(f"Clipping: {clip}")
    print()
    lr_schedule = optax.cosine_decay_schedule(
        init_value=starting_lr, decay_steps=decay_steps, alpha=alpha
    )

    return optax.chain(
        optax.clip_by_global_norm(clip),
        optax.rmsprop(learning_rate=lr_schedule),
    )


def log_hyperparameters(config):
    log_filepath = f"logs/{config['run_id']}_hyperparameters.json"
    pathlib.Path(log_filepath).parent.mkdir(parents=True, exist_ok=True)

    with open(log_filepath, "w") as log_file:
        json.dump(config, log_file, indent=2)
    print(f"Hyperparameters logged to {log_filepath}")


def main(config):
    actor_critic_config = config_utils.create_actor_critic_config(config)
    training_config = config_utils.create_training_config(config)

    model = initialize_model(config["MODEL_NAME"])
    approximator = func_approx.actor_critic_function_aproximation(
        model, func_approx.into_tensor_representation, game.action_to_num
    )
    optimizer = setup_optimizer(
        config["STARTING_LR"], config["DECAY_STEPS"], config["ALPHA"], config["CLIP"]
    )
    key = jax.random.PRNGKey(config["SEED"])
    loop_fn, init_training_state_fn = actor_critic.get_algo_functions(
        optimizer, approximator, actor_critic_config
    )

    training.train(
        key,
        loop_fn,
        init_training_state_fn,
        approximator,
        optimizer,
        training_config,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train A2C model.")
    parser.add_argument(
        "--config", required=True, help="Path to the configuration file"
    )
    args = parser.parse_args()

    config = config_utils.load_config(args.config)

    log_hyperparameters(config)

    main(config)
