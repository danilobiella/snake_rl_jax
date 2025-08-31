import glob
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns  # type: ignore
from matplotlib.lines import Line2D

GRID_SIZE = 5
CONFIDENCE_INTERVAL_PERCENTILE = 100

figures_folder = f"figures/{GRID_SIZE}x{GRID_SIZE}"
results_folder = f"results/{GRID_SIZE}x{GRID_SIZE}"
pathlib.Path(figures_folder).parent.mkdir(parents=True, exist_ok=True)


data = pd.concat(
    [
        pd.read_csv(f).assign(name=".".join(f.split("/")[-1].split(".")[:-1]))
        for f in glob.glob(f"{results_folder}/*.csv")
        if "grid" in f
    ]
).sort_values("name")


# Plot Time

sns.relplot(
    x="time",
    y="snake_lenght",
    hue="name",
    kind="line",
    data=data,
    aspect=2,
    facet_kws=dict(sharey=False),
    estimator="median",
    errorbar=("pi", CONFIDENCE_INTERVAL_PERCENTILE),
    alpha=0.6,
)

plt.plot(
    [0, data["time"].max()],
    [GRID_SIZE * GRID_SIZE, GRID_SIZE * GRID_SIZE],
    "k--",
    lw=1,
)

plt.tight_layout()
plt.xlabel("Time (s)")
plt.ylabel("Snake lenght")
plt.savefig(f"{figures_folder}_time.png", dpi=200)
plt.close()

# Plot n_steps

sns.relplot(
    x="n_steps",
    y="snake_lenght",
    hue="name",
    kind="line",
    data=data,
    aspect=2,
    facet_kws=dict(sharey=False),
    estimator="median",
    errorbar=("pi", CONFIDENCE_INTERVAL_PERCENTILE),
    alpha=0.6,
)

plt.plot(
    [0, data["n_steps"].max()],
    [GRID_SIZE * GRID_SIZE, GRID_SIZE * GRID_SIZE],
    "k--",
    lw=1,
)

plt.tight_layout()
plt.xlabel("Steps")
plt.ylabel("Snake lenght")
plt.savefig(f"{figures_folder}_n_steps.png", dpi=300)
plt.close()

# %% Histogram of performance of last step

last_times = data.groupby("name")["time"].transform("max")
last_rows = data[data["time"] == last_times]

plt.figure(figsize=(8, 6))
for name, group in last_rows.groupby("name"):
    x = np.sort(group["snake_lenght"])
    y = np.arange(1, len(x) + 1) / len(x)
    plt.step(x, y, label=name, linewidth=2.0, alpha=0.7)

plt.xlabel("Snake length")
plt.ylabel("Cumulative probability")
plt.title("ECDF of snake length at last time per run")
plt.legend()
plt.xlim(0, 26)
plt.savefig(f"{figures_folder}_last_step_histogram.png", dpi=300)
