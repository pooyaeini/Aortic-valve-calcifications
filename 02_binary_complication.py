"""
02_binary_complication.py
=========================
SECONDARY ENDPOINT: VARC-2 composite of MAJOR periprocedural complications
(disabling stroke, life-threatening bleeding, major vascular complication,
conversion to surgery, coronary obstruction, AKIN >=2, or in-hospital death).

Interpretable ML classifiers using PRE-PROCEDURAL predictors (clinical, echo,
MDCT anatomy, cusp-specific calcium topography, EuroSCORE II, planned device),
benchmarked against EuroSCORE II.

Models
------
* EuroSCORE II (logistic, single covariate)        -- clinical benchmark
* L2 logistic regression                            -- regularized linear
* Random forest                                     -- non-linear ensemble
* Histogram gradient boosting (LightGBM-style)      -- non-linear boosting

Validation
----------
* 20 x repeated, stratified 5-fold cross-validation.
* Discrimination AUROC + AUPRC; overall Brier; calibration slope/intercept;
  sensitivity/specificity/PPV/NPV at the Youden threshold.
* Honest out-of-fold probabilities for calibration, decision-curve analysis
  and SHAP explainability (SHAP computed on a full-data boosting model).
* Subgroup performance by sex and by age group (fairness check).

Outputs -> ../figures, ../tables, ../results
"""
from __future__ import annotations
import warnings, json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss,
                             roc_curve, precision_recall_curve, confusion_matrix)

import common as C

warnings.filterwarnings("ignore")
np.random.seed(C.RANDOM_STATE)
plt.rcParams.update({"font.family": "serif", "font.size": 11, "axes.grid": True,
                     "grid.alpha": 0.3, "figure.dpi": 150, "savefig.bbox": "tight"})
N_REPEATS, N_SPLITS = 20, 5

try:
    import shap
    HAVE_SHAP = True
except Exception:
    HAVE_SHAP = False


def build_models():
    return {
        "EuroSCORE II": ("logit", ["ES2"]),
        "L2 logistic regression": ("logit", C.FEATURES_FULL),
        "Random forest": ("rf", C.FEATURES_FULL),
        "Histogram gradient boosting": ("hgb", C.FEATURES_FULL),
    }


def make_estimator(kind):
    if kind == "logit":
        return LogisticRegression(penalty="l2", C=1.0, max_iter=5000,
                                  class_weight="balanced")
    if kind == "rf":
        return RandomForestClassifier(
            n_estimators=600, min_samples_leaf=8, max_features="sqrt",
            class_weight="balanced_subsample", n_jobs=-1, random_state=C.RANDOM_STATE)
    if kind == "hgb":
        return HistGradientBoostingClassifier(
            learning_rate=0.05, max_depth=3, max_iter=300, l2_regularization=1.0,
            min_samples_leaf=20, random_state=C.RANDOM_STATE)
    raise ValueError(kind)


def preprocess_fit(Xtr, features, scale):
    cont = [f for f in features if f not in C.BINARY_VARS]
    binv = [f for f in features if f in C.BINARY_VARS]
    imp_c = SimpleImputer(strategy="median").fit(Xtr[cont]) if cont else None
    imp_b = SimpleImputer(strategy="most_frequent").fit(Xtr[binv]) if binv else None
    scaler = StandardScaler().fit(imp_c.transform(Xtr[cont])) if (cont and scale) else None
    return dict(cont=cont, binv=binv, imp_c=imp_c, imp_b=imp_b, scaler=scaler)


def preprocess_apply(X, tf):
    parts, cols = [], []
    if tf["cont"]:
        Xc = tf["imp_c"].transform(X[tf["cont"]])
        if tf["scaler"] is not None:
            Xc = tf["scaler"].transform(Xc)
        parts.append(Xc); cols += tf["cont"]
    if tf["binv"]:
        parts.append(tf["imp_b"].transform(X[tf["binv"]])); cols += tf["binv"]
    return pd.DataFrame(np.hstack(parts), columns=cols, index=X.index)


def metrics_at_threshold(y, p, thr):
    yhat = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    ppv = tp / (tp + fp) if (tp + fp) else np.nan
    npv = tn / (tn + fn) if (tn + fn) else np.nan
    return sens, spec, ppv, npv


def calibration_slope_intercept(y, p):
    from sklearn.linear_model import LogisticRegression as LR
    lp = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    lr = LR(penalty=None, solver="lbfgs", max_iter=1000).fit(lp.reshape(-1, 1), y)
    slope = float(lr.coef_[0][0]); intercept = float(lr.intercept_[0])
    return slope, intercept


def run_cv(df):
    y = C.make_composite_complication(df).values
    models = build_models()
    rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS,
                                   random_state=C.RANDOM_STATE)
    rows = []
    oof_sum = {m: np.zeros(len(df)) for m in models}
    oof_cnt = {m: np.zeros(len(df)) for m in models}
    for fold, (tr, te) in enumerate(rskf.split(df, y)):
        for name, (kind, feats) in models.items():
            Xall = C.feature_frame(df, feats)
            scale = (kind == "logit")
            tf = preprocess_fit(Xall.iloc[tr], feats, scale)
            Xtr = preprocess_apply(Xall.iloc[tr], tf)
            Xte = preprocess_apply(Xall.iloc[te], tf)
            est = make_estimator(kind).fit(Xtr.values, y[tr])
            p = est.predict_proba(Xte.values)[:, 1]
            oof_sum[name][te] += p; oof_cnt[name][te] += 1
            rec = dict(model=name, fold=fold,
                       auroc=roc_auc_score(y[te], p),
                       auprc=average_precision_score(y[te], p),
                       brier=brier_score_loss(y[te], p))
            rows.append(rec)
    cvdf = pd.DataFrame(rows)
    oof = {m: np.where(oof_cnt[m] > 0, oof_sum[m] / np.maximum(oof_cnt[m], 1), np.nan)
           for m in models}
    return cvdf, oof, y


def summarize(cvdf, oof, y):
    def ci(x):
        x = np.asarray(x); return (x.mean(), np.percentile(x, 2.5), np.percentile(x, 97.5))
    out = []
    for m, g in cvdf.groupby("model"):
        row = {"model": m}
        for met in ["auroc", "auprc", "brier"]:
            mean, lo, hi = ci(g[met]); row[met] = mean
            row[met + "_lo"] = lo; row[met + "_hi"] = hi
        # threshold metrics from pooled out-of-fold probs (Youden)
        p = oof[m]; valid = ~np.isnan(p)
        fpr, tpr, thr = roc_curve(y[valid], p[valid])
        j = np.argmax(tpr - fpr); t_opt = thr[j]
        sens, spec, ppv, npv = metrics_at_threshold(y[valid], p[valid], t_opt)
        slope, intercept = calibration_slope_intercept(y[valid], p[valid])
        row.update(dict(threshold=t_opt, sens=sens, spec=spec, ppv=ppv, npv=npv,
                        cal_slope=slope, cal_intercept=intercept))
        out.append(row)
    order = ["EuroSCORE II", "L2 logistic regression", "Random forest",
             "Histogram gradient boosting"]
    return pd.DataFrame(out).set_index("model").reindex(order).reset_index()


# --------------------------- figures --------------------------------------- #
def fig_roc(oof, y, path):
    fig, ax = plt.subplots(figsize=(5.8, 5.6))
    colors = {"EuroSCORE II": "#1b4965", "L2 logistic regression": "#5fa8d3",
              "Random forest": "#2a9d8f", "Histogram gradient boosting": "#c1121f"}
    for m, p in oof.items():
        v = ~np.isnan(p); fpr, tpr, _ = roc_curve(y[v], p[v])
        auc = roc_auc_score(y[v], p[v])
        ax.plot(fpr, tpr, color=colors.get(m, None), lw=2, label=f"{m} (AUC {auc:.2f})")
    ax.plot([0, 1], [0, 1], ls="--", color="grey")
    ax.set_xlabel("1 - specificity"); ax.set_ylabel("Sensitivity")
    ax.set_title("ROC: major periprocedural complication"); ax.legend(fontsize=8, loc="lower right")
    fig.savefig(path); fig.savefig(path.replace(".png", ".pdf")); plt.close(fig)


def fig_pr(oof, y, path):
    fig, ax = plt.subplots(figsize=(5.8, 5.6))
    prev = y.mean()
    colors = {"EuroSCORE II": "#1b4965", "L2 logistic regression": "#5fa8d3",
              "Random forest": "#2a9d8f", "Histogram gradient boosting": "#c1121f"}
    for m, p in oof.items():
        v = ~np.isnan(p); pr, rc, _ = precision_recall_curve(y[v], p[v])
        ap = average_precision_score(y[v], p[v])
        ax.plot(rc, pr, color=colors.get(m, None), lw=2, label=f"{m} (AP {ap:.2f})")
    ax.axhline(prev, ls="--", color="grey", label=f"Prevalence {prev:.2f}")
    ax.set_xlabel("Recall (sensitivity)"); ax.set_ylabel("Precision (PPV)")
    ax.set_title("Precision-recall: complication"); ax.legend(fontsize=8, loc="upper right")
    fig.savefig(path); fig.savefig(path.replace(".png", ".pdf")); plt.close(fig)


def fig_calibration(p, y, path, label):
    v = ~np.isnan(p); p = p[v]; y = y[v]
    bins = np.quantile(p, np.linspace(0, 1, 11)); bins[0] = 0; bins[-1] = 1.0001
    idx = np.digitize(p, bins) - 1
    xs, ys = [], []
    for b in range(10):
        m = idx == b
        if m.sum() > 0:
            xs.append(p[m].mean()); ys.append(y[m].mean())
    fig, ax = plt.subplots(figsize=(5.6, 5.6))
    ax.plot([0, 1], [0, 1], ls="--", color="grey")
    ax.plot(xs, ys, "o-", color="#c1121f")
    mx = max(0.5, max(xs + ys) * 1.1)
    ax.set_xlim(0, mx); ax.set_ylim(0, mx)
    ax.set_xlabel("Predicted probability"); ax.set_ylabel("Observed frequency")
    ax.set_title(f"Calibration: {label}")
    fig.savefig(path); fig.savefig(path.replace(".png", ".pdf")); plt.close(fig)


def fig_dca(oof_best, oof_es2, y, path):
    v = ~np.isnan(oof_best) & ~np.isnan(oof_es2)
    yb = y[v]; pb = oof_best[v]; pe = oof_es2[v]; n = len(yb); prev = yb.mean()
    ths = np.linspace(0.02, 0.6, 50)
    def nb(p):
        out = []
        for pt in ths:
            f = p >= pt
            tp = ((f) & (yb == 1)).sum(); fp = ((f) & (yb == 0)).sum()
            out.append(tp / n - fp / n * (pt / (1 - pt)))
        return np.array(out)
    nb_all = prev - (1 - prev) * (ths / (1 - ths))
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    ax.plot(ths, nb(pb), color="#c1121f", lw=2, label="Calcium-topography ML model")
    ax.plot(ths, nb(pe), color="#1b4965", lw=2, ls="-.", label="EuroSCORE II")
    ax.plot(ths, nb_all, color="grey", ls="--", lw=1.2, label="Treat all")
    ax.axhline(0, color="black", lw=1, label="Treat none")
    ax.set_ylim(-0.05, max(0.05, prev * 1.1))
    ax.set_xlabel("Threshold probability"); ax.set_ylabel("Net benefit")
    ax.set_title("Decision-curve analysis: complication"); ax.legend(fontsize=9)
    fig.savefig(path); fig.savefig(path.replace(".png", ".pdf")); plt.close(fig)


def shap_analysis(df, feats, y, path_bee, path_bar):
    Xall = C.feature_frame(df, feats)
    tf = preprocess_fit(Xall, feats, scale=False)
    X = preprocess_apply(Xall, tf)
    est = make_estimator("hgb").fit(X.values, y)
    Xlab = X.copy(); Xlab.columns = [C.PRETTY.get(c, c) for c in X.columns]
    if not HAVE_SHAP:
        # permutation-importance fallback
        from sklearn.inspection import permutation_importance
        r = permutation_importance(est, X.values, y, n_repeats=20,
                                   random_state=C.RANDOM_STATE, scoring="roc_auc")
        imp = pd.Series(r.importances_mean, index=Xlab.columns).sort_values()
        fig, ax = plt.subplots(figsize=(7.4, 6.2))
        top = imp.tail(18)
        ax.barh(range(len(top)), top.values, color="#1b4965")
        ax.set_yticks(range(len(top))); ax.set_yticklabels(top.index, fontsize=9)
        ax.set_xlabel("Mean AUROC decrease (permutation)")
        ax.set_title("Feature importance: complication model")
        fig.savefig(path_bar); fig.savefig(path_bar.replace(".png", ".pdf")); plt.close(fig)
        return imp
    explainer = shap.TreeExplainer(est)
    sv = explainer.shap_values(X.values)
    if isinstance(sv, list):
        sv = sv[1]
    plt.figure(figsize=(7.6, 6.4))
    shap.summary_plot(sv, Xlab, show=False, max_display=18)
    plt.title("SHAP summary: major periprocedural complication")
    plt.tight_layout(); plt.savefig(path_bee); plt.savefig(path_bee.replace(".png", ".pdf")); plt.close()
    plt.figure(figsize=(7.4, 6.2))
    shap.summary_plot(sv, Xlab, plot_type="bar", show=False, max_display=18)
    plt.title("Mean absolute SHAP: complication model")
    plt.tight_layout(); plt.savefig(path_bar); plt.savefig(path_bar.replace(".png", ".pdf")); plt.close()
    imp = pd.Series(np.abs(sv).mean(0), index=Xlab.columns).sort_values(ascending=False)
    return imp


def subgroup(df, feats, y, oof_best):
    """AUROC by sex and age group using out-of-fold predictions."""
    v = ~np.isnan(oof_best)
    p = oof_best[v]; yy = y[v]
    sub = df.loc[v]
    rows = []
    def auc_ci(yv, pv, n_boot=1000):
        if len(np.unique(yv)) < 2:
            return (np.nan, np.nan, np.nan)
        rng = np.random.RandomState(C.RANDOM_STATE); boots = []
        for _ in range(n_boot):
            idx = rng.randint(0, len(yv), len(yv))
            if len(np.unique(yv[idx])) < 2:
                continue
            boots.append(roc_auc_score(yv[idx], pv[idx]))
        return (roc_auc_score(yv, pv), np.percentile(boots, 2.5), np.percentile(boots, 97.5))
    groups = {
        "Overall": np.ones(len(sub), bool),
        "Female": sub["sex_fem"].values == 1,
        "Male": sub["sex_fem"].values == 0,
        "Age < 82 y": sub["age"].values < 82,
        "Age >= 82 y": sub["age"].values >= 82,
    }
    for g, m in groups.items():
        a, lo, hi = auc_ci(yy[m], p[m])
        rows.append(dict(subgroup=g, n=int(m.sum()), events=int(yy[m].sum()),
                         auroc=a, lo=lo, hi=hi))
    return pd.DataFrame(rows)


def fig_subgroup(sg, path):
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ypos = np.arange(len(sg))[::-1]
    ax.errorbar(sg["auroc"], ypos, xerr=[sg["auroc"] - sg["lo"], sg["hi"] - sg["auroc"]],
                fmt="s", color="#1b4965", ecolor="#5fa8d3", capsize=3)
    ax.axvline(0.5, ls="--", color="grey")
    ax.set_yticks(ypos); ax.set_yticklabels(sg["subgroup"])
    ax.set_xlabel("AUROC (95% bootstrap CI)"); ax.set_xlim(0.4, 1.0)
    ax.set_title("Subgroup discrimination (fairness check)")
    for yv, a, n, e in zip(ypos, sg["auroc"], sg["n"], sg["events"]):
        ax.text(0.42, yv, f"n={n}, ev={e}", fontsize=8, va="center")
    fig.savefig(path); fig.savefig(path.replace(".png", ".pdf")); plt.close(fig)


def main():
    df = C.load_clean()
    print("[complication] running repeated CV ...")
    cvdf, oof, y = run_cv(df)
    res = summarize(cvdf, oof, y)
    res.to_csv(os.path.join(C.TAB_DIR, "table_complication_performance.csv"), index=False)
    cvdf.to_csv(os.path.join(C.RES_DIR, "complication_cv_folds.csv"), index=False)
    print(res.round(3).to_string(index=False))

    cand = res[res.model != "EuroSCORE II"].sort_values("auroc", ascending=False)
    best = cand.iloc[0]["model"]
    print("[complication] best model:", best)

    fig_roc(oof, y, os.path.join(C.FIG_DIR, "fig_complication_roc.png"))
    fig_pr(oof, y, os.path.join(C.FIG_DIR, "fig_complication_pr.png"))
    fig_calibration(oof[best], y, os.path.join(C.FIG_DIR, "fig_complication_calibration.png"), best)
    fig_dca(oof[best], oof["EuroSCORE II"], y, os.path.join(C.FIG_DIR, "fig_complication_dca.png"))

    imp = shap_analysis(df, C.FEATURES_FULL, y,
                        os.path.join(C.FIG_DIR, "fig_complication_shap_beeswarm.png"),
                        os.path.join(C.FIG_DIR, "fig_complication_shap_bar.png"))
    imp.to_csv(os.path.join(C.RES_DIR, "complication_shap_importance.csv"))

    sg = subgroup(df, C.FEATURES_FULL, y, oof[best])
    sg.to_csv(os.path.join(C.TAB_DIR, "table_complication_subgroup.csv"), index=False)
    fig_subgroup(sg, os.path.join(C.FIG_DIR, "fig_complication_subgroup.png"))

    with open(os.path.join(C.RES_DIR, "complication_summary.json"), "w") as f:
        json.dump({"best_model": best, "prevalence": float(y.mean()),
                   "n_events": int(y.sum()), "have_shap": HAVE_SHAP,
                   "cv": res.round(4).to_dict(orient="records")}, f, indent=2)
    print("[complication] done. best AUROC = %.3f" %
          res.set_index("model").loc[best, "auroc"])


if __name__ == "__main__":
    main()
