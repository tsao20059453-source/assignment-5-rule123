# assignment-5-combined-only.py
# Combined false-alarm rate: R1 OR R2(Zone A same side) OR Trend-6
# 輸出：assignment5_any_of_three_summary.csv + assignment5_any_of_three_only.png

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view

def Phi(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

# 單點/單窗機率
alpha1_point = 2.0 * (1.0 - Phi(3.0))              # R1: |Z|>3
pA_one_side  = Phi(3.0) - Phi(2.0)                 # Zone A (單側): 2<=Z<3
alpha2_win3  = 2.0 * (3*(pA_one_side**2)*(1-pA_one_side) + pA_one_side**3)  # R2 每個3點窗
alpha3_win6  = 2.0 / math.factorial(6)             # Trend-6 每個6點窗 = 2/6!

def theory_any(n: int) -> float:
    # 獨立近似
    prod = (1.0 - alpha1_point)**n
    if n >= 3: prod *= (1.0 - alpha2_win3)**(n - 2)
    if n >= 6: prod *= (1.0 - alpha3_win6)**(n - 5)
    return 1.0 - prod

def simulate_any(n: int, reps: int = 50_000, seed: int = 2025) -> float:
    rng = np.random.default_rng(seed + n + reps)
    Z = rng.normal(0.0, 1.0, size=(reps, n))

    r1 = (np.abs(Z) > 3.0).any(axis=1)

    if n >= 3:
        win3 = sliding_window_view(Z, 3, axis=1)
        posA = (win3 > 2.0) & (win3 < 3.0)
        negA = (win3 < -2.0) & (win3 > -3.0)
        r2 = (posA.sum(axis=2) >= 2).any(axis=1) | (negA.sum(axis=2) >= 2).any(axis=1)
    else:
        r2 = np.zeros(reps, dtype=bool)

    if n >= 6:
        win6 = sliding_window_view(Z, 6, axis=1)
        inc  = (win6[...,1:] > win6[...,:-1]).all(axis=2)
        dec  = (win6[...,1:] < win6[...,:-1]).all(axis=2)
        r6 = (inc | dec).any(axis=1)
    else:
        r6 = np.zeros(reps, dtype=bool)

    return float((r1 | r2 | r6).mean())

if __name__ == "__main__":
    ns = [1, 5, 10, 20, 50, 100]
    rows = []
    for n in ns:
        rows.append({
            "n": n,
            "Theoretical_AnyOfThree": theory_any(n),
            "Simulated_AnyOfThree": simulate_any(n),
        })
    df = pd.DataFrame(rows)
    df.to_csv("assignment5_any_of_three_summary.csv", index=False)
    print(df.to_string(index=False))

    # 單一圖：理論 vs 模擬（三規則聯合）
    plt.figure(figsize=(7,4.5))
    plt.plot(df["n"], df["Theoretical_AnyOfThree"], marker="o", linewidth=2, label="Theoretical")
    plt.plot(df["n"], df["Simulated_AnyOfThree"],   marker="o", linewidth=2, label="Simulation")
    plt.xlabel("n (number of points)")
    plt.ylabel("Probability (any of three rules)")
    plt.title("Combined false-alarm rate: R1 OR R2 (Zone A) OR Trend-6")
    plt.legend()
    plt.tight_layout()
    plt.savefig("assignment5_any_of_three_only.png", dpi=200)
    plt.close()
