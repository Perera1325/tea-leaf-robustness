"""Leakage-safe split. Expects: data/<domain>/<class>/<group>__<name>.jpg
domain = estate / dataset / phone (e.g. mendeley, kaggle, own_phone_estateA)
group  = plant/leaf id (photos of the same leaf share a group, so they never straddle train/test)
Outputs splits.csv with columns: path, class, domain, group, split
Protocol A (in-domain): GroupShuffleSplit 70/15/15 inside the source domain
Protocol B (cross-domain): source domain -> train/val ; every OTHER domain -> test_<domain>
Augment ONLY inside the training loop, never on disk before splitting.
"""
import sys, pathlib, pandas as pd
from sklearn.model_selection import GroupShuffleSplit

root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "data")
source = sys.argv[2] if len(sys.argv) > 2 else "mendeley"
rows = []
for p in root.rglob("*"):
    if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
        domain, cls = p.parts[-3], p.parts[-2]
        group = p.stem.split("__")[0] if "__" in p.stem else p.stem  # no group id => each image its own group (WEAK, warn)
        rows.append(dict(path=str(p), cls=cls, domain=domain, group=f"{domain}/{cls}/{group}"))
df = pd.DataFrame(rows)
df["split"] = ""
src = df[df.domain == source]
g1 = GroupShuffleSplit(1, test_size=0.30, random_state=42)
tr, rest = next(g1.split(src, groups=src.group))
rest_df = src.iloc[rest]
g2 = GroupShuffleSplit(1, test_size=0.50, random_state=42)
va, te = next(g2.split(rest_df, groups=rest_df.group))
df.loc[src.index[tr], "split"] = "train"
df.loc[rest_df.index[va], "split"] = "val"
df.loc[rest_df.index[te], "split"] = "test_indomain"
for d in df.domain.unique():
    if d != source:
        df.loc[df.domain == d, "split"] = f"test_{d}"
df.to_csv("splits.csv", index=False)
print(df.groupby(["split", "cls"]).size().unstack(fill_value=0))
assert not set(df[df.split=="train"].group) & set(df[df.split=="test_indomain"].group), "LEAK"
