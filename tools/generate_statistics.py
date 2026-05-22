import pandas as pd
from scipy.stats import ttest_ind

df = pd.read_csv("C:/Documents/Doutoramento/CEA Projects/ucea/tools/comparison_outputs/data_all.csv")

roof = df[df["oven_roof_detected"] == 1]["error_cea_oven"]
no_roof = df[df["oven_roof_detected"] == 0]["error_cea_oven"]

print("Roof detected")
print("N:", len(roof))
print("Mean:", roof.mean())
print("Std:", roof.std())
print("Median:", roof.median())

print("\nNo roof detected")
print("N:", len(no_roof))
print("Mean:", no_roof.mean())
print("Std:", no_roof.std())
print("Median:", no_roof.median())

tstat, pval = ttest_ind(roof, no_roof, equal_var=False)

print("\nWelch t-test p-value:", pval)