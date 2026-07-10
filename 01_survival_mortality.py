"""
01_survival_mortality.py
========================
PRIMARY ENDPOINT: mid-term all-cause mortality after TAVI.

Interpretable machine-learning survival models integrating cusp-specific
aortic-valve and LVOT calcium topography, benchmarked against EuroSCORE II.

Models
------
* EuroSCORE II (Cox, single covariate)            clinical benchmark
* Penalized Cox (Coxnet, elastic net)             regularized linear survival
* Gradient-Boosted Survival (Cox loss)            non-linear boosting
* Random Survival Forest (RSF)                     non-linear ensemble
* RSF without calcium topography                   incremental-value comparator

Validation
----------
* 10 x repeated, event-stratified 5-fold cross-validation (50 folds).
* Discrimination Harrell C-index (per fold, mean 95% CI), time-dependent AUC at
  1/2/3 years, integrated Brier score.
* Honest out-of-fold 2-year mortality risk for calibration, decision-curve
  analysis and risk-group Kaplan-Meier.
* Global explainability C-index permutation importance.

Chunked execution (2-core sandbox, per-call time limit)
-------------------------------------------------------
    python 01_survival_mortality.py chunk <rep_start> <rep_end>   # runs repeats
    python 01_survival_mortality.py aggregate                     # tables+figures
Each chunk appends per-fold metrics to results/survival_cv_folds.csv and updates
out-of-fold predictions in results/survival_oof.json.
"""
from __future__ import annotations
import warnings, json, os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sksurv.util import Surv
from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
from sksurv.metrics import (concordance_index_censored, cumulative_dynamic_auc,
                            integrated_brier_score)

import common as C

warnings.filterwarnings("ignore")
np.random.seed(C.RANDOM_STATE)
plt.rcParams.update({"font.family": "serif", "font.size": 11, "axes.grid": True,
                     "grid.alpha": 0.3, "figure.dpi": 150, "savefig.bbox": "tight"})

HORIZONS = [1.0, 2.0, 3.0]
PRIMARY_HORIZON = 2.0
N_REPEATS = 10
N_SPLITS = 5
FOLD_CSV = os.path.join(C.RES_DIR, "survival_cv_folds.csv")
OOF_JSON = os.path.join(C.RES_DIR, "survival_oof.json")

MODEL_ORDER = ["EuroSCORE II", "Penalized Cox", "Gradient-boosted survival",
               "Random survival forest", "RSF without calcium"]


def build_models():
    return {
        "EuroSCORE II": ("cox", ["ES2"]),
        "Penalized Cox": ("coxnet", C.FEATURES_FULL),
        "Gradient-boosted survival": ("gbs", C.FEATURES_FULL),
        "Random survival forest": ("rsf", C.FEATURES_FULL),
        "RSF without calcium": ("rsf", C.FEATURES_NO_CALCIUM),
    }


def make_estimator(kind):
    if kind == "cox":
        return CoxPHSurvivalAnalysis(alpha=1e-4)
    if kind == "coxnet":
        return CoxnetSurvivalAnalysis(l1_ratio=0.5, alpha_min_ratio=0.01,
                                      fit_baseline_model=True, max_iter=100000)
    if kind == "rsf":
        return RandomSurvivalForest(n_estimators=300, min_samples_leaf=15,
                                    min_samples_split=30, max_features="sqrt",
                                    n_jobs=-1, random_state=C.RANDOM_STATE)
    if kind == "gbs":
        return GradientBoostingSurvivalAnalysis(
            n_estimators=300, learning_rate=0.05, max_depth=2, subsample=0.8,
            min_samples_leaf=15, random_state=C.RANDOM_STATE)
    raise ValueError(kind)


def preprocess_fit(Xtr, features):
    cont = [f for f in features if f not in C.BINARY_VARS]
    binv = [f for f in features if f in C.BINARY_VARS]
    imp_c = SimpleImputer(strategy="median").fit(Xtr[cont]) if cont else None
    imp_b = SimpleImputer(strategy="most_frequent").fit(Xtr[binv]) if binv else None
    scaler = StandardScaler().fit(imp_c.transform(Xtr[cont])) if cont else None
    return dict(cont=cont, binv=binv, imp_c=imp_c, imp_b=imp_b, scaler=scaler)


def preprocess_apply(X, tf):
    parts, cols = [], []
    if tf["cont"]:
        Xc = tf["scaler"].transform(tf["imp_c"].transform(X[tf["cont"]]))
        parts.append(Xc); cols += tf["cont"]
    if tf["binv"]:
        parts.append(tf["imp_b"].transform(X[tf["binv"]])); cols += tf["binv"]
    return pd.DataFrame(np.hstack(parts), columns=cols, index=X.index)


def surv_prob_at(model, X, times):
    try:
        fns = model.predict_survival_function(X)
        return np.vstack([[fn(t) for t in times] for fn in fns])
    except Exception:
        return None


def get_data():
    df = C.load_clean()
    mask, time, event = C.make_survival_endpoint(df)
    dfx = df.loc[mask].reset_index(drop=True)
    return dfx, time[mask], event[mask].astype(bool)


# --------------------------------------------------------------------------- #
# Chunked CV
# --------------------------------------------------------------------------- #
def load_oof(n):
    if os.path.exists(OOF_JSON):
        with open(OOF_JSON) as f:
            d = json.load(f)
        return ({m: np.array(d["sum"][m]) for m in MODEL_ORDER},
                {m: np.array(d["cnt"][m]) for m in MODEL_ORDER},
                set(tuple(x) for x in d.get("done", [])))
    return ({m: np.zeros(n) for m in MODEL_ORDER},
            {m: np.zeros(n) for m in MODEL_ORDER}, set())


def save_oof(osum, ocnt, done):
    with open(OOF_JSON, "w") as f:
        json.dump({"sum": {m: osum[m].tolist() for m in MODEL_ORDER},
                   "cnt": {m: ocnt[m].tolist() for m in MODEL_ORDER},
                   "done": sorted(list(done))}, f)


def run_chunk(rep_start, rep_end):
    dfx, time, event = get_data()
    y = Surv.from_arrays(event=event, time=time)
    models = build_models()
    osum, ocnt, done = load_oof(len(dfx))
    rows = []
    for rep in range(rep_start, rep_end):
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                              random_state=C.RANDOM_STATE + rep)
        for fold, (tr, te) in enumerate(skf.split(dfx, event.astype(int))):
            if (rep, fold) in done:
                continue
            ytr, yte = y[tr], y[te]
            valid_h = [h for h in HORIZONS if h < min(time[te].max(), time[tr].max())]
            for name, (kind, feats) in models.items():
                Xall = C.feature_frame(dfx, feats)
                tf = preprocess_fit(Xall.iloc[tr], feats)
                Xtr = preprocess_apply(Xall.iloc[tr], tf)
                Xte = preprocess_apply(Xall.iloc[te], tf)
                est = make_estimator(kind)
                try:
                    est.fit(Xtr.values, ytr)
                except Exception:
                    continue
                risk = est.predict(Xte.values)
                cidx = concordance_index_censored(yte["event"], yte["time"], risk)[0]
                rec = dict(model=name, rep=rep, fold=fold, cindex=cidx)
                if valid_h:
                    try:
                        auc, _ = cumulative_dynamic_auc(ytr, yte, risk, valid_h)
                        for h, a in zip(valid_h, np.atleast_1d(auc)):
                            rec["auc_%dy" % int(h)] = float(a)
                    except Exception:
                        pass
                    sp = surv_prob_at(est, Xte.values, valid_h)
                    if sp is not None:
                        try:
                            rec["ibrier"] = float(integrated_brier_score(ytr, yte, sp, valid_h))
                        except Exception:
                            pass
                        if PRIMARY_HORIZON in valid_h:
                            j = valid_h.index(PRIMARY_HORIZON)
                            osum[name][te] += (1.0 - sp[:, j]); ocnt[name][te] += 1
                rows.append(rec)
            done.add((rep, fold))
        print("  finished repeat", rep)
    # append fold metrics
    newdf = pd.DataFrame(rows)
    if os.path.exists(FOLD_CSV) and len(newdf):
        newdf.to_csv(FOLD_CSV, mode="a", header=False, index=False)
    elif len(newdf):
        newdf.to_csv(FOLD_CSV, index=False)
    save_oof(osum, ocnt, done)
    print("chunk %d-%d done; total folds recorded=%d" % (rep_start, rep_end, len(done)))


# --------------------------------------------------------------------------- #
# Aggregation, permutation importance, figures
# --------------------------------------------------------------------------- #
def summarize(cvdf):
    def ci(x):
        x = pd.Series(x).dropna().values
        return (np.mean(x), np.percentile(x, 2.5), np.percentile(x, 97.5)) if len(x) else (np.nan,)*3
    out = []
    for m in MODEL_ORDER:
        g = cvdf[cvdf.model == m]
        row = {"model": m}
        for met in ["cindex", "auc_1y", "auc_2y", "auc_3y", "ibrier"]:
            if met in g and g[met].notna().any():
                mean, lo, hi = ci(g[met]); row[met] = mean
                row[met+"_lo"] = lo; row[met+"_hi"] = hi
        out.append(row)
    return pd.DataFrame(out)


def permutation_importance_cindex(dfx, time, event, kind, feats, n_repeat=8):
    y = Surv.from_arrays(event=event, time=time)
    Xall = C.feature_frame(dfx, feats)
    tf = preprocess_fit(Xall, feats); X = preprocess_apply(Xall, tf)
    # lighter forest purely for permutation ranking (keeps the call within limits)
    est = RandomSurvivalForest(n_estimators=150, min_samples_leaf=15,
                               min_samples_split=30, max_features="sqrt",
                               n_jobs=-1, random_state=C.RANDOM_STATE).fit(X.values, y)
    base = concordance_index_censored(event, time, est.predict(X.values))[0]
    rng = np.random.RandomState(C.RANDOM_STATE); imp = {}
    for j, col in enumerate(X.columns):
        drops = []
        for _ in range(n_repeat):
            Xp = X.values.copy(); rng.shuffle(Xp[:, j])
            drops.append(base - concordance_index_censored(event, time, est.predict(Xp))[0])
        imp[col] = float(np.mean(drops))
    return pd.Series(imp).sort_values(ascending=False), float(base)


def fig_cindex(res, path):
    r = res.dropna(subset=["cindex"]).copy()
    r = r.set_index("model").reindex(MODEL_ORDER).dropna(subset=["cindex"]).reset_index()
    fig, ax = plt.subplots(figsize=(7.2, 4.2)); ypos = np.arange(len(r))
    ax.errorbar(r["cindex"], ypos, xerr=[r["cindex"]-r["cindex_lo"], r["cindex_hi"]-r["cindex"]],
                fmt="o", color="#1b4965", ecolor="#5fa8d3", capsize=3, ms=7)
    ax.set_yticks(ypos); ax.set_yticklabels(r["model"]); ax.axvline(0.5, ls="--", color="grey")
    ax.set_xlabel("Harrell C-index (mean, 95% CI across 50 CV folds)")
    ax.set_title("Discrimination for mid-term all-cause mortality"); ax.set_xlim(0.45, 0.85)
    for y0, v in zip(ypos, r["cindex"]): ax.text(v, y0+0.16, "%.3f" % v, ha="center", fontsize=9)
    fig.savefig(path); fig.savefig(path.replace(".png", ".pdf")); plt.close(fig)


def fig_importance(imp, path, topn=18):
    top = imp.head(topn)[::-1]
    labels = [C.PRETTY.get(k, k) for k in top.index]
    colors = ["#c1121f" if k in C.CALCIUM else "#1b4965" for k in top.index]
    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    ax.barh(range(len(top)), top.values, color=colors)
    ax.set_yticks(range(len(top))); ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Mean decrease in C-index when permuted")
    ax.set_title("Global permutation importance (random survival forest)")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#c1121f", label="Calcium topography"),
                       Patch(color="#1b4965", label="Other predictor")],
              loc="lower right", fontsize=8)
    fig.savefig(path); fig.savefig(path.replace(".png", ".pdf")); plt.close(fig)


def fig_km(oof, time, event, path):
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import multivariate_logrank_test
    v = ~np.isnan(oof); r = oof[v]; t = time[v]; e = event[v].astype(int)
    terts = pd.qcut(r, 3, labels=["Low", "Intermediate", "High"])
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    col = {"Low": "#2a9d8f", "Intermediate": "#e9c46a", "High": "#c1121f"}
    kmf = KaplanMeierFitter()
    for g in ["Low", "Intermediate", "High"]:
        mm = terts == g; kmf.fit(t[mm], e[mm], label="%s risk (n=%d)" % (g, mm.sum()))
        kmf.plot_survival_function(ax=ax, ci_show=True, color=col[g], lw=2)
    lr = multivariate_logrank_test(t, terts, e)
    ax.set_xlabel("Years after TAVI"); ax.set_ylabel("Survival probability")
    ax.set_title("Survival by model-predicted risk tertile"); ax.set_ylim(0, 1.02)
    ax.text(0.02, 0.06, "log-rank p < 0.001" if lr.p_value < 1e-3 else "log-rank p = %.3f" % lr.p_value,
            transform=ax.transAxes, fontsize=10)
    fig.savefig(path); fig.savefig(path.replace(".png", ".pdf")); plt.close(fig)
    return float(lr.p_value)


def fig_calibration(oof, time, event, path, horizon=PRIMARY_HORIZON):
    from lifelines import KaplanMeierFitter
    v = ~np.isnan(oof); r = oof[v]; t = time[v]; e = event[v].astype(int)
    dec = pd.qcut(r, 10, labels=False, duplicates="drop")
    obs, pred = [], []
    for d0 in np.unique(dec):
        mm = dec == d0; kmf = KaplanMeierFitter().fit(t[mm], e[mm])
        try: s = float(kmf.predict(horizon))
        except Exception: s = np.nan
        obs.append(1 - s); pred.append(float(np.mean(r[mm])))
    fig, ax = plt.subplots(figsize=(5.6, 5.6)); ax.plot([0, 1], [0, 1], ls="--", color="grey")
    ax.scatter(pred, obs, color="#1b4965", s=45, zorder=3)
    mx = max(0.6, np.nanmax(obs + pred) * 1.1); ax.set_xlim(0, mx); ax.set_ylim(0, mx)
    ax.set_xlabel("Predicted %d-year mortality" % int(horizon))
    ax.set_ylabel("Observed %d-year mortality (1-KM)" % int(horizon))
    ax.set_title("Calibration (out-of-fold, risk deciles)")
    fig.savefig(path); fig.savefig(path.replace(".png", ".pdf")); plt.close(fig)
    return np.array(pred), np.array(obs)


def fig_dca(oof_best, oof_es2, time, event, path, horizon=PRIMARY_HORIZON):
    from lifelines import KaplanMeierFitter
    def nb(risk, ths):
        v = ~np.isnan(risk); r = risk[v]; t = time[v]; e = event[v].astype(int); n = len(r); out = []
        for pt in ths:
            fl = r >= pt
            if fl.sum() == 0: out.append(0.0); continue
            kmf = KaplanMeierFitter().fit(t[fl], e[fl])
            try: ev = 1 - float(kmf.predict(horizon))
            except Exception: ev = np.nan
            out.append(ev*fl.sum()/n - (1-ev)*fl.sum()/n*(pt/(1-pt)))
        return np.array(out)
    kmf = KaplanMeierFitter().fit(time, event.astype(int)); ev_all = 1 - float(kmf.predict(horizon))
    ths = np.linspace(0.02, 0.5, 40); nb_all = ev_all - (1-ev_all)*(ths/(1-ths))
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    ax.plot(ths, nb(oof_best, ths), color="#c1121f", lw=2, label="Calcium-topography ML model")
    ax.plot(ths, nb(oof_es2, ths), color="#1b4965", lw=2, ls="-.", label="EuroSCORE II")
    ax.plot(ths, nb_all, color="grey", ls="--", lw=1.2, label="Treat all")
    ax.axhline(0, color="black", lw=1, label="Treat none")
    ax.set_xlabel("Threshold probability (%d-year mortality)" % int(horizon))
    ax.set_ylabel("Net benefit"); ax.set_title("Decision-curve analysis")
    ax.set_ylim(-0.05, max(0.05, ev_all*1.1)); ax.legend(fontsize=9)
    fig.savefig(path); fig.savefig(path.replace(".png", ".pdf")); plt.close(fig)


def calibration_slope(oof, time, event):
    from lifelines import CoxPHFitter
    v = ~np.isnan(oof); r = np.clip(oof[v], 1e-4, 1-1e-4)
    lp = np.log(r/(1-r)); lp = (lp-lp.mean())/lp.std()
    d = pd.DataFrame({"lp": lp, "t": time[v], "e": event[v].astype(int)})
    try:
        return float(CoxPHFitter().fit(d, "t", "e").params_["lp"])
    except Exception:
        return np.nan


def _best_model(res):
    cand = res[~res.model.isin(["EuroSCORE II", "RSF without calcium"])].dropna(subset=["cindex"])
    return cand.sort_values("cindex", ascending=False).iloc[0]["model"]


def _load_oof_dict():
    with open(OOF_JSON) as f:
        oofd = json.load(f)
    osum = {m: np.array(oofd["sum"][m]) for m in MODEL_ORDER}
    ocnt = {m: np.array(oofd["cnt"][m]) for m in MODEL_ORDER}
    return {m: np.where(ocnt[m] > 0, osum[m]/np.maximum(ocnt[m], 1), np.nan) for m in MODEL_ORDER}


def aggregate_figs():
    """Fast: summary table, C-index figure, OOF KM/calibration/DCA, summary json."""
    dfx, time, event = get_data()
    res = summarize(pd.read_csv(FOLD_CSV))
    res.to_csv(os.path.join(C.TAB_DIR, "table_survival_performance.csv"), index=False)
    best = _best_model(res)
    print("[survival] best ML model:", best)
    fig_cindex(res, os.path.join(C.FIG_DIR, "fig_survival_cindex.png"))
    oof = _load_oof_dict()
    lp = fig_km(oof[best], time, event, os.path.join(C.FIG_DIR, "fig_survival_km_risk.png"))
    fig_calibration(oof[best], time, event, os.path.join(C.FIG_DIR, "fig_survival_calibration.png"))
    fig_dca(oof[best], oof["EuroSCORE II"], time, event,
            os.path.join(C.FIG_DIR, "fig_survival_dca.png"))
    slope = calibration_slope(oof[best], time, event)
    summary = {"best_model": best, "logrank_p_riskgroups": lp,
               "calibration_slope_std": slope, "n_analysable": int(len(dfx)),
               "n_events": int(event.sum()), "cv": res.round(4).to_dict(orient="records")}
    with open(os.path.join(C.RES_DIR, "survival_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("[survival] figures + tables complete.")


def aggregate_importance():
    dfx, time, event = get_data()
    res = summarize(pd.read_csv(FOLD_CSV)); best = _best_model(res)
    kind, feats = build_models()[best]
    imp, base = permutation_importance_cindex(dfx, time, event, kind, feats)
    imp.to_csv(os.path.join(C.RES_DIR, "survival_permutation_importance.csv"))
    fi