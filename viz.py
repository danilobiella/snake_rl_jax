import glob
import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns  # type: ignore

GRID_SIZE = 5
figures_folder = f"figures/{GRID_SIZE}x{GRID_SIZE}"
results_folder = f"results/{GRID_SIZE}x{GRID_SIZE}"
pathlib.Path(figures_folder).parent.mkdir(parents=True, exist_ok=True)


data = pd.concat(
    [
        pd.read_csv(f).assign(name=f.split("/")[-1].split(".")[0])
        for f in glob.glob(f"{results_folder}/*.csv")
    ]
).sort_values("name")


def sum_total_time_taken(df):
    result = (
        df.drop_duplicates(subset=["n_game", "n_frames", "name"])
        .set_index("n_frames")
        .sort_index()
        .time.cumsum()
        .rename("cum_time")
    )
    return result


if len(data.name.unique()) > 1:
    df_total_time_taken = (
        data.groupby("name")
        .apply(sum_total_time_taken)
        .reset_index()
        .rename(columns={"level_1": "n_frames"})
    )
else:
    df_total_time_taken = sum_total_time_taken(data)
    df_total_time_taken = df_total_time_taken.reset_index()
    df_total_time_taken["name"] = data.name.unique()[0]


data = data.merge(df_total_time_taken, on=["name", "n_frames"], how="left")


# Plot Time

sns.relplot(
    x="cum_time",
    y="snake_lenght",
    hue="name",
    kind="line",
    data=data,
    aspect=2,
    facet_kws=dict(sharey=False),
    estimator="median",
    errorbar=("pi", 75),
    alpha=0.6,
)

plt.plot(
    [0, data["cum_time"].max()],
    [GRID_SIZE * GRID_SIZE, GRID_SIZE * GRID_SIZE],
    "k--",
    lw=1,
)

plt.tight_layout()
plt.savefig(f"{figures_folder}_time.png", dpi=200)
plt.close()

# Plot n_frames

sns.relplot(
    x="n_frames",
    y="snake_lenght",
    hue="name",
    kind="line",
    data=data,
    aspect=2,
    facet_kws=dict(sharey=False),
    estimator="median",
    errorbar=("pi", 75),
    alpha=0.6,
)

plt.plot(
    [0, data["n_frames"].max()],
    [GRID_SIZE * GRID_SIZE, GRID_SIZE * GRID_SIZE],
    "k--",
    lw=1,
)

plt.tight_layout()
plt.savefig(f"{figures_folder}_n_frames.png", dpi=300)
plt.close()
