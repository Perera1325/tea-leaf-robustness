"""Append the external TEST domains to splits.csv (idempotent).
Never touches train / val / test_indomain."""
import pandas as pd, pathlib
df = pd.read_csv("splits.csv")
df = df[df.domain == "mendeley"]                      # drop any old external rows
EXT = {"hf_tea": "data/hf_tea", "mendeley_ext": "data/mendeley_ext"}
rows = []
for dom, root in EXT.items():
    for p in sorted(pathlib.Path(root).rglob("*.jpg")):
        rows.append(dict(path=p.as_posix(), cls=p.parent.name, domain=dom,
                         group=f"{dom}/{p.parent.name}/{p.stem}", split=f"test_{dom}"))
df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
df.to_csv("splits.csv", index=False)
open("classes.txt", "w").write("\n".join(sorted(df[df.domain == "mendeley"].cls.unique())))
print(df.split.value_counts())
