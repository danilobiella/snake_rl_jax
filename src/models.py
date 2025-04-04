import haiku as hk
import jax


def get_model(model_name, grid_size):
    if grid_size == 5:
        if model_name == "nn":
            return get_actor_critic_nn_5x5()
        if model_name == "linear":
            return get_actor_critic_linear_model()
        raise ValueError(f"Model {model_name} not supported for grid size {grid_size}")
    raise ValueError(f"Gird size {grid_size} not supported")


def get_actor_critic_linear_model():
    def forward(x):
        actor = hk.Linear(64, name="actor_hidden")(x)
        actor = jax.nn.relu(actor)
        actor = hk.Linear(4, name="actor_output")(actor)

        critic = hk.Linear(64, name="critic_hiden")(x)
        critic = jax.nn.relu(critic)
        critic = hk.Linear(1, name="critic_output")(critic)

        return (actor, critic)

    return hk.transform(forward)


def get_actor_critic_nn_5x5():
    def forward(x):
        x = hk.Conv2D(
            32, kernel_shape=(3, 3), stride=1, padding="VALID", name="first_conv_layer"
        )(x)
        x = jax.nn.relu(x)

        x = hk.Conv2D(
            64, kernel_shape=(3, 3), stride=1, padding="VALID", name="second_conv_layer"
        )(x)
        x = jax.nn.relu(x)

        x = hk.Flatten(preserve_dims=-3, name="flatten")(x)

        x = hk.Linear(256, "dense_layer")(x)
        x = jax.nn.relu(x)

        return _actor_critic_output(x)

    return hk.transform(forward)


def _actor_critic_output(x):
    actor = hk.Linear(4, name="actor")(x)
    critic = hk.Linear(1, name="critic")(x)
    return (actor, critic)
