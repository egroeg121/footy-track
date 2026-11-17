"""Dev-iteration comparison on the cards every run answered (the 4 few-shot cards excluded)."""
import score

SHOTS = {"cfea4830d5", "bcf380f1fe", "a07b286173", "de62e8402f"}
RUNS = ["dev_A_v1_still-strip", "dev_B_v2_still-wide-strip", "dev_C_v3_meta", "dev_D_v3_meta_4shot",
        "dev_E_v3_still-wide_nomotion"]
data = {r: {x["id"]: x for x in score.load(r)[0] if x["id"] not in SHOTS} for r in RUNS}
common = set.intersection(*(set(d) for d in data.values()))
print(f"common cards: {len(common)}")
print("| run | acc4 | binary kappa (ball-present) | ball prec | not_ball prec | not_ball recall | mean cost $/card |")
print("|---|---|---|---|---|---|---|")
for r in RUNS:
    rows = [data[r][i] for i in sorted(common)]
    n = len(rows)
    acc = sum(x["pred"] == x["george"] for x in rows)
    pres = lambda v: "p" if v in ("ball", "box_off") else ("a" if v == "not_ball" else "u")
    b = [(pres(x["george"]), pres(x["pred"])) for x in rows if x["george"] != "unsure" and x["pred"] != "unsure"]
    bp = [x for x in rows if x["pred"] == "ball"]
    npd = [x for x in rows if x["pred"] == "not_ball"]
    gn = [x for x in rows if x["george"] == "not_ball"]
    print(f"| {r} | {score.fmt(acc, n)} | {score.kappa(b):.3f} | "
          f"{score.fmt(sum(x['george'] == 'ball' for x in bp), len(bp))} | "
          f"{score.fmt(sum(x['george'] == 'not_ball' for x in npd), len(npd))} | "
          f"{score.fmt(sum(x['pred'] == 'not_ball' for x in gn), len(gn))} | "
          f"{sum(x['cost'] for x in rows) / n:.4f} |")
