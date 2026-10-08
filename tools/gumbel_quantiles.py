"""
Fit a Gumbel (EV1) distribution to a series of annual peak discharges and print
the design discharge for each return period used by the simulator.

Paste the printed values into "Manual input" in index.html, or use the
mean and Cv in the "Gumbel distribution" option.

Input: a CSV file with one column of annual maximum discharges (m3/s).
The column named "peak_m3s" is used if present, otherwise the last column.

Usage:
    python tools/gumbel_quantiles.py data/example_annual_peaks.csv
    python tools/gumbel_quantiles.py my_peaks.csv --method moments
"""
import argparse
import csv
import math
import sys

T_LIST = [2, 5, 10, 25, 50, 100, 200]
EULER = 0.5772156649


def read_series(path):
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    header, body = rows[0], rows[1:]
    try:
        col = header.index("peak_m3s")
    except ValueError:
        col = len(header) - 1
    vals = []
    for r in body:
        try:
            vals.append(float(r[col]))
        except (ValueError, IndexError):
            continue
    if len(vals) < 10:
        sys.exit("Need at least 10 annual peaks for a meaningful fit.")
    return vals


def fit_moments(x):
    n = len(x)
    mean = sum(x) / n
    sd = math.sqrt(sum((v - mean) ** 2 for v in x) / (n - 1))
    beta = sd * math.sqrt(6) / math.pi
    return mean - EULER * beta, beta


def fit_lmoments(x):
    xs = sorted(x)
    n = len(xs)
    l1 = sum(xs) / n
    b1 = sum((i / (n - 1)) * v for i, v in enumerate(xs)) / n
    l2 = 2 * b1 - l1
    beta = l2 / math.log(2)
    return l1 - EULER * beta, beta


def quantile(xi, beta, T):
    return xi - beta * math.log(-math.log(1 - 1 / T))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv_file")
    ap.add_argument("--method", choices=["lmoments", "moments"], default="lmoments")
    a = ap.parse_args()

    x = read_series(a.csv_file)
    n = len(x)
    mean = sum(x) / n
    sd = math.sqrt(sum((v - mean) ** 2 for v in x) / (n - 1))
    xi, beta = (fit_lmoments if a.method == "lmoments" else fit_moments)(x)

    print(f"Years of record: {n}")
    print(f"Mean = {mean:.1f} m3/s, SD = {sd:.1f} m3/s, Cv = {sd / mean:.2f}")
    print(f"Gumbel ({a.method}): location = {xi:.1f}, scale = {beta:.1f}\n")
    print(f"{'T (years)':>10} {'Q (m3/s)':>10}")
    for T in T_LIST:
        print(f"{T:>10} {quantile(xi, beta, T):>10.0f}")
    print("\nPaste the Q values into 'Manual input' in index.html.")
