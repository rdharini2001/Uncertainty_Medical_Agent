"""
Pipeline:  tools (plain Python)  ->  Investigator  ->  Visual  ->  Skeptic  ->  Judge
Baseline:  same tool outputs     ->  one single agent gives the verdict

"""

import argparse
import json
import os
import re
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MODEL = os.environ.get("GEMMA_MODEL", "gemma-4-31b-it")


def call_gemma(prompt, image_paths=None, temperature=0.3, retries=4):
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    parts = []
    for p in image_paths or []:  # images first: recommended for Gemma 4
        with open(p, "rb") as f:
            parts.append(types.Part.from_bytes(data=f.read(), mime_type="image/png"))
    parts.append(prompt)
    for attempt in range(retries):
        try:
            resp = client.models.generate_content(
                model=MODEL,
                contents=parts,
                config=types.GenerateContentConfig(temperature=temperature),
            )
            return resp.text
        except Exception as e:  # rate limits on free tier: back off and retry
            wait = 10 * (attempt + 1)
            print(f"  [LLM error: {e}] retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError("LLM call failed after retries")


def extract_json(text):
    """Pull the first {...} block out of a model reply."""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return {"parse_error": True, "raw": text}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {"parse_error": True, "raw": text}


# ---------------------------------------------------------------------------
# Tools (deterministic)
# ---------------------------------------------------------------------------
def get_features(df, site, outcome, covariates):
    meta = {site, outcome, *covariates}
    return [c for c in df.columns if c not in meta and pd.api.types.is_numeric_dtype(df[c])]


def tool_site_classifier(df, feats, site):
    """Can the hospital be predicted from the features? Balanced acc vs chance."""
    X, y = df[feats].values, df[site].astype(str).values
    n_sites = len(np.unique(y))
    k = min(5, pd.Series(y).value_counts().min())
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    scores = cross_val_score(clf, X, y, cv=StratifiedKFold(k, shuffle=True, random_state=0),
                             scoring="balanced_accuracy")
    return {"n_sites": int(n_sites), "chance_level": round(1 / n_sites, 3),
            "balanced_accuracy_mean": round(float(scores.mean()), 3),
            "balanced_accuracy_std": round(float(scores.std()), 3)}


def tool_feature_stats(df, feats, site, top_k=10):
    """
Kruskal-Wallis across sites + eta^2
    """
    rows = []
    groups = df[site].astype(str)
    for f in feats:
        vals = [df.loc[groups == g, f].dropna().values for g in groups.unique()]
        h, p = stats.kruskal(*vals)
        grand = df[f].mean()
        ss_between = sum(len(v) * (v.mean() - grand) ** 2 for v in vals)
        ss_total = ((df[f] - grand) ** 2).sum()
        rows.append({"feature": f, "kruskal_p": p, "eta2_site": ss_between / ss_total if ss_total else 0})
    res = pd.DataFrame(rows).sort_values("eta2_site", ascending=False)
    n = len(res)
    res["p_bonferroni"] = (res["kruskal_p"] * n).clip(upper=1)
    summary = {
        "n_features": n,
        "n_significant_bonferroni": int((res["p_bonferroni"] < 0.05).sum()),
        "top_features": res.head(top_k).round(4).to_dict("records"),
    }
    return summary, res


def tool_pca_plot(df, feats, site, out_path):
    X = StandardScaler().fit_transform(df[feats].values)
    pca = PCA(n_components=2).fit(X)
    Z = pca.transform(X)
    plt.figure(figsize=(6, 5))
    for g in sorted(df[site].astype(str).unique()):
        m = df[site].astype(str).values == g
        plt.scatter(Z[m, 0], Z[m, 1], s=12, alpha=0.7, label=f"site {g}")
    ev = pca.explained_variance_ratio_
    plt.xlabel(f"PC1 ({ev[0]:.0%})"); plt.ylabel(f"PC2 ({ev[1]:.0%})")
    plt.title("PCA of radiomics features, coloured by site"); plt.legend()
    plt.tight_layout(); plt.savefig(out_path, dpi=110); plt.close()
    return out_path


def tool_boxplots(df, stats_df, site, out_path, k=6):
    top = stats_df.head(k)["feature"].tolist()
    fig, axes = plt.subplots(2, 3, figsize=(11, 6))
    for ax, f in zip(axes.ravel(), top):
        df.boxplot(column=f, by=site, ax=ax)
        ax.set_title(f, fontsize=9); ax.set_xlabel("site")
    plt.suptitle("Top site-associated features (by eta^2)"); plt.tight_layout()
    plt.savefig(out_path, dpi=110); plt.close()
    return out_path



def run_tools(df, site, outcome, covariates, workdir):
    os.makedirs(workdir, exist_ok=True)
    feats = get_features(df, site, outcome, covariates)
    clf = tool_site_classifier(df, feats, site)
    fstats, fstats_df = tool_feature_stats(df, feats, site)
    pca_png = tool_pca_plot(df, feats, site, os.path.join(workdir, "pca.png"))
    box_png = tool_boxplots(df, fstats_df, site, os.path.join(workdir, "boxplots.png"))
    return {"classifier": clf, "feature_stats": fstats, 
            "pca_png": pca_png, "box_png": box_png}


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------
def investigator(t):
    prompt = f"""You are the INVESTIGATOR in a team auditing a multi-site radiomics dataset for batch (site/scanner) effects.
Below are results of analyses already run on the data.

Site classifier (can features predict the hospital?):
{json.dumps(t['classifier'], indent=1)}

Per-feature site association (Kruskal-Wallis; eta2 = share of feature variance explained by site; top features shown):
{json.dumps(t['feature_stats'], indent=1, default=float)}

Report in JSON:
{{"site_predictable": "yes|no|unclear",
  "strength": "none|weak|moderate|strong",
  "affected_features": [list of feature names you consider site-affected],
  "global_vs_local": "global (many features) | local (few features) | none",
  "evidence": "2-4 sentences citing the numbers"}}"""
    return extract_json(call_gemma(prompt))


def visual(t):
    prompt = """You are the VISUAL REVIEWER. Image 1: PCA scatter of radiomics features coloured by hospital site.
Image 2: box plots of the most site-associated features, grouped by site.
Judge ONLY from what you see. Report in JSON:
{"pca_separates_sites": "yes|partly|no",
 "boxplots_show_site_shift": "yes|partly|no",
 "notes": "2-3 sentences describing what you see (which sites, which features, how strong)"}"""
    return extract_json(call_gemma(prompt, [t["pca_png"], t["box_png"]]))


def skeptic(t, inv, vis):
    prompt = f"""You are the SKEPTIC. Your job is to argue against the conclusion "these site differences are a technical
scanner/protocol artifact". Consider whether hospitals simply have different patients (real biology) (confounding case).
Critically examine the visual findings.

- Does PCA separation necessarily indicate scanner effects?
- Could the observed separation be explained by differences in patient populations?
- Do the boxplots show consistent shifts across hospitals, or could they reflect biological differences?
- Are the visual findings consistent with the investigator's statistical findings?

Do not assume that agreement between the investigator and visual reviewer
proves the existence of technical batch effects.

Investigator's findings:
{json.dumps(inv, indent=1)}

Visual reviewer's findings:
{json.dumps(vis, indent=1)}

Report in JSON:
{{"confounding_present": "yes|no|unclear",
  "confounded_variables": [list],
  "alternative_explanation": "1-3 sentences: could the site differences be biological? how plausible?",
  "risk_if_harmonized_naively": "1-2 sentences: what would ComBat without protecting these variables do?"}}"""
    return extract_json(call_gemma(prompt))


def judge(inv, vis, skp):
    prompt = f"""You are the JUDGE. Combine three independent reports about a multi-site radiomics dataset.
When reports disagree, lower your confidence and say why.

INVESTIGATOR: {json.dumps(inv)}
VISUAL REVIEWER: {json.dumps(vis)}
SKEPTIC: {json.dumps(skp)}

Give the final verdict in JSON:
{{"batch_effect": "yes|no|uncertain",
  "affected_features": [list],
  "confounded": "yes|no|uncertain",
  "recommendation": "e.g. harmonize with ComBat preserving <vars> | no harmonization needed | collect more info",
  "confidence": a number between 0 and 1,
  "agreement": "full|partial|conflict",
  "reasoning": "3-5 sentences"}}"""
    return extract_json(call_gemma(prompt))


def single_agent_baseline(t):
    prompt = f"""You are auditing a multi-site radiomics dataset for batch (site/scanner) effects.
Images: PCA coloured by site, and box plots of the most site-associated features.

Site classifier: {json.dumps(t['classifier'])}
Per-feature stats: {json.dumps(t['feature_stats'], default=float)}

Give the final verdict in JSON:
{{"batch_effect": "yes|no|uncertain", "affected_features": [list], "confounded": "yes|no|uncertain",
  "recommendation": "...", "confidence": 0-1, "reasoning": "3-5 sentences"}}"""
    return extract_json(call_gemma(prompt, [t["pca_png"], t["box_png"]]))


def run_team(t):
    inv = investigator(t)
    vis = visual(t)
    skp = skeptic(t, inv, vis)
    verdict = judge(inv, vis, skp)
    return {"investigator": inv, "visual": vis, "skeptic": skp, "judge": verdict}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--site", default="site")
    ap.add_argument("--outcome", default="outcome")
    ap.add_argument("--covariates", nargs="*", default=[])
    ap.add_argument("--mode", choices=["team", "baseline", "both"], default="team")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    name = os.path.splitext(os.path.basename(args.csv))[0]
    workdir = os.path.join(args.out, name)
    t = run_tools(df, args.site, args.outcome, args.covariates, workdir)

    result = {"dataset": name, "tools": {k: v for k, v in t.items() if not k.endswith("_png")}}
    if args.mode in ("team", "both"):
        result["team"] = run_team(t)
    if args.mode in ("baseline", "both"):
        result["baseline"] = single_agent_baseline(t)

    with open(os.path.join(workdir, "result.json"), "w") as f:
        json.dump(result, f, indent=2, default=float)
    print(json.dumps({k: result[k] for k in result if k in ("team", "baseline")}, indent=2, default=float))


if __name__ == "__main__":
    main()