"""Evaluate the trained PPO gripper co-design policy across grasping tasks.

Loads a saved checkpoint, samples the learned design policy for egg and
cylinder tasks, and writes the resulting design distribution for analysis.
Set GRIPPER_PPO_MODEL_PATH to evaluate a different checkpoint."""

from pathlib import Path
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from gripper_rl_env import GripperEnv


# SETTINGS
THIS_DIR = Path(__file__).resolve().parent  # .../Gripper/RL

MODEL_PATH = Path(os.environ["GRIPPER_PPO_MODEL_PATH"]).expanduser().resolve() if os.environ.get("GRIPPER_PPO_MODEL_PATH") else THIS_DIR / "models" / "ywk5xiqs" / "final_model.zip"
OUTPUT_CSV = THIS_DIR / "design_distribution.csv"
SAMPLES_PER_TASK = 10  
SEED = 42
np.random.seed(SEED)

# INIT ENV + MODEL
env = GripperEnv(mode="compact", verbose=False)
model = PPO.load(MODEL_PATH)

print(f"✅ Loaded model from {MODEL_PATH}")
print(f"Object types: {env.object_types}")
print(f"Weight choices: {env.weight_choices}")

# Define evaluation tasks
tasks = [("egg", i) for i in [5, 6, 7]] + [("cylinder", i) for i in [2, 3, 4]]


# Collect design samples
results = []

for obj_type, weight_num in tasks:

    # --- Fix Task Parameters ---
    env.current_object_type = obj_type
    env.current_weight_num = weight_num
    env.weight_value = float(weight_num)

    # --- Initialize observation ---
    obs = np.zeros(7 + len(env.object_types) + 1, dtype=np.float32)
    obs[-(len(env.object_types) + 1):-1] = env._one_hot_encode(obj_type, env.object_types)
    obs[-1] = env.weight_value 

    
    print(f"\n🎯 Evaluating task: {obj_type}, weight={weight_num}")


    for _ in range(SAMPLES_PER_TASK):
        action, _ = model.predict(obs, deterministic=False)
        decoded = env._decode_action(action)
        # convert init_grasp_height by adding the object radius

        if obj_type == "egg":
            obj_radius = 30  # mm
            decoded[2] += obj_radius
        elif obj_type == "cylinder":
            obj_radius = 32/2  # mm
            decoded[2] += obj_radius
        results.append({
            "object_type": obj_type,
            "weight_num": weight_num,
            "init_grasp_height": decoded[0],
            "maximum_pressure": decoded[1],
            "grasp_radius_distance": decoded[2],
            "inclination_angle": decoded[3],
            "inner_radius": decoded[4],
            "average_radius": decoded[5],
            "module_length": decoded[6],
            "module_number": decoded[7],
            "actuator_number": decoded[8],
        })
        

# Convert to DataFrame
df = pd.DataFrame(results)
df.to_csv(OUTPUT_CSV, index=False)
print(f"\n✅ Saved design distribution to {OUTPUT_CSV}")
print(df.head())

# Quick visualization example

fig, axes = plt.subplots(3, 3, figsize=(12, 6))


param_map = {
    "init_grasp_height": "grasping height (mm)",
    "maximum_pressure": "maximum pressure (kPa)",
    "grasp_radius_distance": "actuator base radius (mm)",
    "inclination_angle": "inclination angle (rad)",
    "inner_radius": "inner radius (mm)",
    "average_radius": "average radius (mm)",
    "module_length": "module length (mm)",
    "module_number": "module number",
    "actuator_number": "actuator number",
}
params = list(param_map.keys())

x_round_digits = [1, 0, 0, 2, 1, 1, 1, 0, 0]  
action_low = np.array([5, 10, 0, 0, 2, 6, 8, 5, 2])
action_high = np.array([15, 15, 10, np.pi/6, 4, 8, 12, 10, 4])
value_min = [5, 10, 15, 0, 2, 6, 8, 5, 2]
value_max = [15, 15, 30, np.pi/6, 4, 8, 12, 10, 4]
xtick_counts = [5, 6, 6, 6, 6, 6, 6, 6, 3]  

for ax, param in zip(axes.flatten(), params):
    idx = params.index(param)
    pmin, pmax = value_min[params.index(param)], value_max[params.index(param)]
    bins = np.linspace(pmin, pmax, 11)  # 10 bins → 11 edges

    # Plot histograms for each object type
    for obj in env.object_types:
        subset = df[df["object_type"] == obj][param]
        ax.hist(subset, bins=bins, alpha=0.6, label=obj, edgecolor="black", linewidth=0.5)

    ax.set_xticks(np.linspace(value_min[idx], value_max[idx], xtick_counts[idx]))
    ax.set_xticklabels(
        [f"{tick:.{x_round_digits[params.index(param)]}f}" for tick in ax.get_xticks()]
    )
    ax.set_title(param_map[param], fontsize=10)
    ax.tick_params(axis='both', labelsize=8)
    ax.legend()

plt.tight_layout()
plt.show()


# not overlapping version
fig, axes = plt.subplots(3, 3, figsize=(12, 6))

param_map = {
    "init_grasp_height": "grasping height (mm)",
    "maximum_pressure": "maximum pressure (kPa)",
    "grasp_radius_distance": "actuator base radius (mm)",
    "inclination_angle": "inclination angle (rad)",
    "inner_radius": "inner radius (mm)",
    "average_radius": "average radius (mm)",
    "module_length": "module length (mm)",
    "module_number": "module number",
    "actuator_number": "actuator number",
}
params = list(param_map.keys())

x_round_digits = [1, 0, 0, 2, 1, 1, 1, 0, 0] 
action_low = np.array([5, 10, 0, 0, 2, 6, 8, 5, 2])
action_high = np.array([15, 15, 10, np.pi/6, 4, 8, 12, 10, 4])

value_min = [5, 10, 15, 0, 2, 6, 8, 5, 2]
value_max = [15, 15, 30, np.pi/6, 4, 8, 12, 10, 4]
xtick_counts = [5, 6, 6, 6, 6, 6, 6, 6, 3] 

for ax, param in zip(axes.flatten(), params):
    idx = params.index(param)

    pmin, pmax = value_min[idx], value_max[idx]
    bins = np.linspace(pmin, pmax, 11)  # 10 bins → 11 edges
    bin_edges = bins
    bin_width = bin_edges[1] - bin_edges[0]
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    obj_types = list(env.object_types)
    n = len(obj_types)
    if n == 0:
        continue

    bar_width = 0.9 * bin_width / n

    for i, obj in enumerate(obj_types):
        subset = df.loc[df["object_type"] == obj, param].dropna().to_numpy()
        counts, _ = np.histogram(subset, bins=bin_edges)

        offset = (i - (n - 1) / 2) * bar_width

        ax.bar(
            bin_centers + offset,
            counts,
            width=bar_width,
            alpha=0.75,
            label=obj,
            edgecolor="black",
            linewidth=0.5,
            align="center",
        )

    ax.set_xticks(np.linspace(value_min[idx], value_max[idx], xtick_counts[idx]))
    ax.set_xticklabels(
        [f"{tick:.{x_round_digits[idx]}f}" for tick in ax.get_xticks()]
    )

    ax.set_title(param_map[param], fontsize=10)
    ax.tick_params(axis="both", labelsize=8)
    ax.legend(fontsize=7)

plt.tight_layout()
plt.show()


fig, axes = plt.subplots(3, 3, figsize=(12, 6))

params = [
    "init_grasp_height", "maximum_pressure", "grasp_radius_distance",
    "inclination_angle", "inner_radius", "average_radius",
    "module_length", "module_number", "actuator_number"
]

groups = [
    ("egg", 5), ("egg", 6), ("egg", 7),
    ("cylinder", 2), ("cylinder", 3), ("cylinder", 4),
]


colors = {
    ("egg", 5): "#1f77b4",  # blue
    ("egg", 6): "#2ca02c",  # green
    ("egg", 7): "#9467bd",  # purple
    ("cylinder", 2): "#d62728",  # red
    ("cylinder", 3): "#ff7f0e",  # orange
    ("cylinder", 4): "#8c564b",  # brown
}

for ax, param in zip(axes.flatten(), params):

    all_vals = df[param].values
    pmin, pmax = value_min[params.index(param)], value_max[params.index(param)]

    # fixed bins
    bins = np.linspace(pmin, pmax, 11)
    bin_width = bins[1] - bins[0]

    n_groups = len(groups)
    group_width = bin_width * 0.9
    bar_width = group_width / n_groups

    offsets = np.linspace(-group_width/2, group_width/2, n_groups)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    for gi, (obj, w) in enumerate(groups):
        subset = df[(df["object_type"] == obj) & (df["weight_num"] == w)][param].values
        counts, _ = np.histogram(subset, bins=bins)

        ax.bar(
            bin_centers + offsets[gi],
            counts,
            width=bar_width,
            color=colors[(obj, w)],
            alpha=0.6,
            label=f"{obj}-{w}"
        )

    ax.set_xticks(np.linspace(value_min[params.index(param)], value_max[params.index(param)], xtick_counts[params.index(param)]))
    ax.set_xticklabels(
        [f"{tick:.{x_round_digits[params.index(param)]}f}" for tick in ax.get_xticks()]
    )
    ax.set_title(param_map[param], fontsize=10)
    ax.tick_params(axis='both', labelsize=8)
    ax.legend(fontsize=7, ncol=2, frameon=True)

plt.tight_layout()
plt.show()
