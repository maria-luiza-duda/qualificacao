import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("experiments.csv")

# ----------------------
# pipeline usage
# ----------------------

pipeline_counts = df["pipeline"].value_counts()

plt.figure()
pipeline_counts.plot(kind="bar")
plt.title("Pipeline Usage")
plt.ylabel("Number of Experiments")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("pipeline_usage.png")

# ----------------------
# guidance scale evolution
# ----------------------

plt.figure()
plt.plot(df["guidance_scale"], marker="o")
plt.title("Guidance Scale Across Experiments")
plt.ylabel("guidance_scale")
plt.xlabel("experiment index")
plt.tight_layout()
plt.savefig("guidance_evolution.png")

# ----------------------
# strength evolution
# ----------------------

plt.figure()
plt.plot(df["strength"], marker="o")
plt.title("Strength Parameter Evolution")
plt.ylabel("strength")
plt.xlabel("experiment index")
plt.tight_layout()
plt.savefig("strength_evolution.png")

# ----------------------
# controlnet scale
# ----------------------

plt.figure()
plt.plot(df["controlnet_scale"], marker="o")
plt.title("ControlNet Scale Evolution")
plt.ylabel("controlnet_scale")
plt.xlabel("experiment index")
plt.tight_layout()
plt.savefig("controlnet_scale_evolution.png")

# ----------------------
# appearance prior
# ----------------------

prior_counts = df["appearance_prior_used"].value_counts()

plt.figure()
prior_counts.plot(kind="bar")
plt.title("Appearance Prior Usage")
plt.ylabel("Number of Experiments")
plt.tight_layout()
plt.savefig("appearance_prior_usage.png")

print("Graphs generated!")