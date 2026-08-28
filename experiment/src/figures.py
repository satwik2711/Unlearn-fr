"""Regenerate figures only from saved tidy artifacts after confirmation."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main():
    d = pd.read_csv("experiment/artifacts/author_effects.csv")
    Path("experiment/artifacts").mkdir(parents=True, exist_ok=True)
    ax = d.plot(kind="bar", x="state", y="mean_delta", yerr="ci_halfwidth")
    plt.tight_layout()
    plt.savefig("experiment/artifacts/main_recovery_plot.png", dpi=180)
    plt.close()


if __name__ == "__main__":
    main()
