"""
03_tables.py
============
Descriptive tables for the manuscript.

* Table 1  Baseline characteristics of the analysis cohort, overall and stratified
           by mid-term vital status (alive vs dead at last follow-up), with
           standardized differences and group comparisons.
* Predictor dictionary (name, domain, definition, type, missingness).
* Endpoint / event-count summary.

Outputs -> ../tables
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from scipy import stats
import common as C


CONT_VARS = ["age", "bmi", "bsa", "creatinine_pre", "crea_clearance", "ef",
             "pht_mmHg", "echo_pre_Pmax", "echo_pre_Pmean", "ava",
             "CT_Annulus_diamMax", "CT_Annulus_diammin", "CT_area",
             "CT_annulus_perimeter", "CT_Distanz_RCA", "Distanz_LCA",
             "oversizing", "eccentricity_index", "DLZ_calcium", "total_calcium",
             "LCC_calcium", "RCC_calcium", "NCC_calcium", "total_calciumLVOT",
             "LCC_calciumLVOT", "RCC_calciumLVOT", "NCC_calciumLVOT",
             "Additive_ES", "Logistic_ES", "ES2", "size"]
CAT_VARS = ["sex_fem", "dialysis_pre", "extracardiac_arteriopathy", "poor_mob",
            "redo", "pre_cabg", "copd", "critical_preop_state", "iddm", "niddm",
            "ccs4", "recent_myocard_infarction", "pht_severe", "pre_pmk",
            "af_paroxysm", "af_pers_perm", "pci_preop", "urgency", "access",
            "device_self_expanding"]


def smd_cont(a, b):
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    sd = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return (a.mean() - b.mean()) / sd if sd else np.nan


def smd_bin(a, b):
    pa, pb = np.nanmean(a), np.nanmean(b)
    sd = np.sqrt((pa * (1 - pa) + pb * (1 - pb)) / 2)
    return (pa - pb) / sd if sd else np.nan


def baseline_table(df):
    mask, time, event = C.make_survival_endpoint(df)
    d = df.loc[mask].copy()
    dead = d[event[mask].astype(bool)]
    alive = d[~event[mask].astype(bool)]
    rows = []
    for v in CONT_VARS:
        allv = pd.to_numeric(d[v], errors="coerce")
        av = pd.to_numeric(alive[v], errors="coerce")
        dv = pd.to_numeric(dead[v], errors="coerce")
        try:
            p = stats.mannwhitneyu(av.dropna(), dv.dropna()).pvalue
        except Exception:
            p = np.nan
        rows.append(dict(
            variable=C.PRETTY.get(v, v),
            overall=f"{allv.mean():.1f} ({allv.std():.1f})",
            alive=f"{av.mean():.1f} ({av.std():.1f})",
            dead=f"{dv.mean():.1f} ({dv.std():.1f})",
            smd=f"{smd_cont(av.values, dv.values):.2f}",
            p=f"{p:.3f}" if p == p else "NA",
            missing=f"{100*allv.isna().mean():.1f}%"))
    for v in CAT_VARS:
        allv = pd.to_numeric(d[v], errors="coerce")
        av = pd.to_numeric(alive[v], errors="coerce")
        dv = pd.to_numeric(dead[v], errors="coerce")
        tbl = pd.crosstab(event[mask].astype(bool), pd.to_numeric(d[v], errors="coerce"))
        try:
            p = stats.chi2_contingency(tbl)[1]
        except Exception:
            p = np.nan
        rows.append(dict(
            variable=C.PRETTY.get(v, v),
            overall=f"{int(np.nansum(allv))} ({100*np.nanmean(allv):.1f}%)",
            alive=f"{int(np.nansum(av))} ({100*np.nanmean(av):.1f}%)",
            dead=f"{int(np.nansum(dv))} ({100*np.nanmean(dv):.1f}%)",
            smd=f"{smd_bin(av.values, dv.values):.2f}",
            p=f"{p:.3f}" if p == p else "NA",
            missing=f"{100*allv.isna().mean():.1f}%"))
    tab = pd.DataFrame(rows)
    tab.columns = ["Variable", f"Overall (n={len(d)})",
                   f"Alive (n={len(alive)})", f"Dead (n={len(dead)})",
                   "SMD", "p", "Missing"]
    return tab


def predictor_dictionary():
    domain = {}
    for f in C.CLINICAL: domain[f] = "Clinical"
    for f in C.ECHO: domain[f] = "Echocardiography"
    for f in C.CT_ANATOMY: domain[f] = "MDCT anatomy"
    for f in C.CALCIUM: domain[f] = "Calcium topography"
    domain["ES2"] = "Surgical risk score"
    for f in ["access", "device_self_expanding", "size"]: domain[f] = "Planned device"
    df = C.load_clean()
    rows = []
    for f in C.FEATURES_FULL:
        col = pd.to_numeric(df.get(f), errors="coerce")
        rows.append(dict(feature=f, label=C.PRETTY.get(f, f),
                         domain=domain.get(f, ""),
                         type="binary" if f in C.BINARY_VARS else "continuous",
                         missing=f"{100*col.isna().mean():.1f}%"))
    return pd.DataFrame(rows)


def endpoint_summary(df):
    mask, time, event = C.make_survival_endpoint(df)
    comp = C.make_composite_complication(df)
    rows = [
        dict(endpoint="Mid-term all-cause mortality (primary)",
             analysable=int(mask.sum()), events=int(event[mask].sum()),
             rate=f"{100*event[mask].mean():.1f}%",
             detail=f"median follow-up {np.nanmedian(time[mask]):.2f} y"),
        dict(endpoint="Composite major periprocedural complication (secondary)",
             analysable=len(df), events=int(comp.sum()),
             rate=f"{100*comp.mean():.1f}%",
             detail="stroke/LT-bleed/vascular/conversion/coronary/AKIN>=2/in-hosp death"),
    ]
    return pd.DataFrame(rows)


def main():
    df = C.load_clean()
    baseline_table(df).to_csv(os.path.join(C.TAB_DIR, "table1_baseline.csv"), index=False)
    predictor_dictionary().to_csv(os.path.join(C.TAB_DIR, "table_predictor_dictionary.csv"), index=False)
    endpoint_summary(df).to_csv(os.path.join(C.TAB_DIR, "table_endpoint_summary.csv"), index=False)
    print("[tables] baseline, predictor dictionary, endpoint summary written.")
    print(endpoint_summary(df).to_string(index=False))


if __name__ == "__main__":
    main()
