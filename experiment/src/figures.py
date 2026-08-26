"""Regenerate figures only from saved CSV artifacts (called after confirmation)."""

import pandas as pd, matplotlib.pyplot as plt
from pathlib import Path


def main():
    d = pd.read_csv("experiment/artifacts/author_effects.csv")
    Path("experiment/artifacts").mkdir(exist_ok=True)
    ax = d.plot(kind="bar", x="state", y="mean_delta", yerr="ci_halfwidth")
    plt.tight_layout()
    plt.savefig("experiment/artifacts/main_recovery_plot.png", dpi=180)
    plt.close()


if __name__ == "__main__":
    main()
