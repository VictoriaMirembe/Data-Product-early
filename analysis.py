"""
UICT Student Success Early-Warning — analysis pipeline
Covers: Descriptive, Diagnostic, Predictive (trained model), Prescriptive.
Run:  python3 analysis.py   -> writes results.json, model.joblib, clean CSVs
"""
import json, re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
import joblib

UP = "./"
CUS = [f"CU{i}" for i in range(1, 7)]
OUT = {}

# ------------------------------------------------------------------ 1. CLEAN
def load_perf(path):
    df = pd.read_csv(UP + path)
    df = df[df.STUDENT_NO.str.startswith("S", na=False)].copy()      # drop "#N/A" rows
    for cu in CUS:
        # "50*" = retake / supplementary mark. Keep the number, flag the retake.
        mk = df[f"{cu}_MK"].astype(str)
        df[f"{cu}_RETAKE"] = mk.str.contains(r"\*").astype(int)
        for part in ["CW", "EX", "MK", "GP"]:
            df[f"{cu}_{part}"] = pd.to_numeric(
                df[f"{cu}_{part}"].astype(str).str.replace("*", "", regex=False), errors="coerce")
    return df

perf = pd.concat([load_perf("DITA student persformance 2020 sem1.csv"),
                  load_perf("DITS student performance 2020 sem1.csv")], ignore_index=True)
perf = perf.drop(columns=["Id", "DataSetId", "SEMESTER"])
n_raw = len(perf)
# Students missing most marks probably dropped out mid-semester -> report them, exclude from model
perf["CUS_SAT"] = perf[[f"{c}_MK" for c in CUS]].notna().sum(axis=1)
dropouts = perf[perf.CUS_SAT < 6]
perf = perf[perf.CUS_SAT == 6].copy()

mk_cols = [f"{c}_MK" for c in CUS]
cw_cols = [f"{c}_CW" for c in CUS]
ex_cols = [f"{c}_EX" for c in CUS]
perf["AVG_MARK"] = perf[mk_cols].mean(axis=1)
perf["CGPA"] = perf[[f"{c}_GP" for c in CUS]].mean(axis=1)
perf["AVG_CW"] = perf[cw_cols].mean(axis=1)
perf["N_FAILS"] = (perf[mk_cols] < 50).sum(axis=1)
perf["N_RETAKES"] = perf[[f"{c}_RETAKE" for c in CUS]].sum(axis=1)
# TARGET: at risk = failed or had to retake any course unit, or CGPA below 3.0 (pass-class boundary)
perf["AT_RISK"] = ((perf.N_FAILS > 0) | (perf.N_RETAKES > 0) | (perf.CGPA < 3.0)).astype(int)
perf.to_csv("clean_performance.csv", index=False)

# ------------------------------------------------------------- 2. DESCRIPTIVE
d = {}
d["students_total"] = int(n_raw)
d["students_complete"] = int(len(perf))
d["incomplete_records"] = int(len(dropouts))
d["avg_mark"] = round(perf.AVG_MARK.mean(), 1)
d["avg_cgpa"] = round(perf.CGPA.mean(), 2)
d["at_risk_rate"] = round(perf.AT_RISK.mean() * 100, 1)
d["by_program"] = perf.groupby("STUDENT_PROGRAM").agg(
    n=("AVG_MARK", "size"), avg_mark=("AVG_MARK", "mean"), cgpa=("CGPA", "mean"),
    at_risk=("AT_RISK", "mean")).round(2).reset_index().to_dict("records")
d["cu_avg"] = {p: perf[perf.STUDENT_PROGRAM == p][mk_cols].mean().round(1).tolist()
               for p in perf.STUDENT_PROGRAM.unique()}
d["cu_cw_ex"] = {"cw": perf[cw_cols].mean().round(1).tolist(), "ex": perf[ex_cols].mean().round(1).tolist()}
bins = [0, 50, 60, 70, 80, 101]; labels = ["<50", "50-59", "60-69", "70-79", "80+"]
d["mark_dist"] = pd.cut(perf[mk_cols].values.ravel(), bins, labels=labels, right=False
                        ).value_counts().reindex(labels).astype(int).tolist()
d["mark_dist_labels"] = labels
d["by_sex"] = perf.groupby("STUDENT_SEX").agg(n=("CGPA", "size"), cgpa=("CGPA", "mean"),
              at_risk=("AT_RISK", "mean")).round(2).reset_index().to_dict("records")
d["by_study_time"] = perf.groupby("STUDY_TIME").agg(n=("CGPA", "size"), cgpa=("CGPA", "mean"),
              at_risk=("AT_RISK", "mean")).round(2).reset_index().to_dict("records")
OUT["descriptive"] = d

# --------------------------------------------------------------- 3. DIAGNOSTIC
g = {}
# Which course unit drives failure? fail/retake count per CU
g["cu_fail_count"] = [int(((perf[f"{c}_MK"] < 50) | (perf[f"{c}_RETAKE"] == 1)).sum()) for c in CUS]
# Correlation: coursework vs final mark (is CW a good early signal?)
g["corr_avgcw_avgmark"] = round(perf.AVG_CW.corr(perf.AVG_MARK), 2)
g["corr_cw_mk_per_cu"] = [round(perf[f"{c}_CW"].corr(perf[f"{c}_MK"]), 2) for c in CUS]
g["corr_matrix"] = perf[mk_cols].corr().round(2).values.tolist()
# Exam vs coursework share: where are marks lost?
g["exam_pct_of_max"] = round(perf[ex_cols].mean().mean() / 60 * 100, 1)   # exam out of 60
g["cw_pct_of_max"] = round(perf[cw_cols].mean().mean() / 40 * 100, 1)     # coursework out of 40
g["risk_vs_ok_cw"] = perf.groupby("AT_RISK").AVG_CW.mean().round(1).to_dict()
g["scatter"] = perf[["AVG_CW", "AVG_MARK", "AT_RISK", "STUDENT_PROGRAM"]].round(1).values.tolist()
OUT["diagnostic"] = g

# --------------------------------------------------------------- 4. PREDICTIVE
# EARLY-WARNING framing: predict risk using ONLY information available BEFORE exams
# (coursework marks + study mode + sex + programme). That's what makes it useful.
X = perf[cw_cols].copy()
X["IS_DITS"] = (perf.STUDENT_PROGRAM == "DITS").astype(int)
X["IS_FEMALE"] = (perf.STUDENT_SEX == "F").astype(int)
for t in ["DAY", "EVENING", "WEEKEND"]:
    X[f"STUDY_{t}"] = (perf.STUDY_TIME == t).astype(int)
y = perf.AT_RISK
feat = list(X.columns)

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
models = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000)),
    "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=4, class_weight="balanced", random_state=42),
}
cv = StratifiedKFold(5, shuffle=True, random_state=42)
p = {"features": feat, "n_train": int(len(X_tr)), "n_test": int(len(X_te)), "models": {}}
for name, m in models.items():
    m.fit(X_tr, y_tr); pr = m.predict(X_te)
    p["models"][name] = {
        "accuracy": round(accuracy_score(y_te, pr), 2), "recall": round(recall_score(y_te, pr), 2),
        "precision": round(precision_score(y_te, pr, zero_division=0), 2), "f1": round(f1_score(y_te, pr), 2),
        "cv_recall": round(cross_val_score(m, X, y, cv=cv, scoring="recall").mean(), 2),
        "cv_accuracy": round(cross_val_score(m, X, y, cv=cv, scoring="accuracy").mean(), 2),
        "confusion": confusion_matrix(y_te, pr).tolist()}
rf = models["Random Forest"].fit(X, y)                      # refit best-explained model on all data
lr = models["Logistic Regression"].fit(X, y)
imp = sorted(zip(feat, rf.feature_importances_), key=lambda t: -t[1])
p["importance"] = [[f, round(float(v), 3)] for f, v in imp]
p["lr_coef"] = dict(zip(feat, np.round(lr[-1].coef_[0], 2).tolist()))
perf["RISK_PROB"] = rf.predict_proba(X)[:, 1]
joblib.dump({"model": rf, "features": feat}, "model.joblib")
OUT["predictive"] = p

# ------------------------------------------------------------ 5. PRESCRIPTIVE
def recommend(r):
    weak = [c for c in CUS if r[f"{c}_CW"] < 25]            # coursework below 25/40 (62.5%)
    if r.RISK_PROB >= 0.7: tier, act = "High", "Assign academic mentor + weekly check-ins before exams"
    elif r.RISK_PROB >= 0.4: tier, act = "Medium", "Targeted revision clinics in weak units"
    else: tier, act = "Low", "Standard support; consider peer-tutor role"
    if weak: act += f"; remedial focus: {', '.join(weak)}"
    return pd.Series({"TIER": tier, "ACTION": act})
perf[["TIER", "ACTION"]] = perf.apply(recommend, axis=1)
rx = {}
rx["tier_counts"] = perf.TIER.value_counts().reindex(["High", "Medium", "Low"]).fillna(0).astype(int).to_dict()
rx["priority_cus"] = [c for c, _ in sorted(zip(CUS, g["cu_fail_count"]), key=lambda t: -t[1])][:2]
rx["watchlist"] = perf.sort_values("RISK_PROB", ascending=False)[
    ["STUDENT_NO", "STUDENT_PROGRAM", "STUDY_TIME", "AVG_CW", "RISK_PROB", "AT_RISK", "TIER", "ACTION"]
].round(2).head(15).values.tolist()
perf.to_csv("student_risk_scores.csv", index=False)
OUT["prescriptive"] = rx

# ---------------------------------------------- 6. LMS USAGE (platform-level)
lms = pd.read_csv(UP + "Learner management system usage.csv", encoding="latin1")
clean = lambda s: re.sub(r"[^\x20-\x7E]", "", str(s)).strip(' "')
lms["ROLE"] = lms.USER_ROLE.map(clean).str.title()
def days(s):
    s = clean(s)
    if s == "Never": return np.nan
    y = re.search(r"(\d+) year", s); dd = re.search(r"(\d+) day", s)
    if not (y or dd or re.search(r"hour|min|sec", s)): return None   # junk/shifted rows
    return (int(y.group(1)) * 365 if y else 0) + (int(dd.group(1)) if dd else 0)
lms["DAYS"] = lms.LAST_LOGIN.map(days)
lms["NEVER"] = lms.LAST_LOGIN.map(clean).eq("Never")
lms = lms[lms.NEVER | lms.DAYS.notna()]
stu = lms[lms.ROLE.str.contains("Student|Participant|Learner")]
L = {"records": int(len(lms)), "learner_records": int(len(stu)),
     "never_pct": round(stu.NEVER.mean() * 100, 1)}
def bucket(r):
    if r.NEVER: return "Never logged in"
    return "< 1 year" if r.DAYS < 365 else ("1-2 years" if r.DAYS < 730 else "2+ years")
stu = stu.assign(B=stu.apply(bucket, axis=1))
order = ["< 1 year", "1-2 years", "2+ years", "Never logged in"]
L["recency"] = stu.B.value_counts().reindex(order).fillna(0).astype(int).tolist(); L["recency_labels"] = order
cu = stu.groupby(stu.COURSE_NAME.map(clean)).agg(n=("NEVER", "size"), never=("NEVER", "mean"))
cu = cu[cu.n >= 30].sort_values("never", ascending=False)
L["course_never"] = [[k, int(r.n), round(r.never * 100, 1)] for k, r in cu.iterrows()][:10]
OUT["lms"] = L

# ------------------------------------------------- 7. DIGITAL SKILLS SURVEY
sv = pd.read_csv(UP + "students_survey_data digital skilling.csv")
sv = sv[sv.INTERNET_ACCESS.isin(["Yes", "No"])]          # drop rows shifted by bad CSV quoting
occ = sv.OCCUPATION.astype(str).str.strip().str.lower()
sv["OCC"] = np.select([occ.str.contains("teach|tutor|lecturer|trainer"),
                       occ.str.contains("stud|vac|s.6|s6|learner|school")], ["Teacher/Trainer", "Student"], "Other/Working")
exp = sv.EXPECTATIONS.astype(str).str.lower()
themes = {"Skills/knowledge": "skill|knowledge|literate|basic|learn", "Certificate": "certif",
          "Job/employment": "job|employ|work|earn|living", "Business": "business|self employ|creat",
          "Cyber security": "cyber|security", "Programming/networking": "program|network|coding"}
S = {"n": int(len(sv)), "gender": sv.GENDER.value_counts().to_dict(),
     "internet_yes_pct": round((sv.INTERNET_ACCESS == "Yes").mean() * 100, 1),
     "occupation": sv.OCC.value_counts().to_dict(),
     "top_districts": sv.DISTRICT.astype(str).str.strip().str.title().value_counts().head(8).to_dict(),
     "expectations": {k: int(exp.str.contains(v).sum()) for k, v in themes.items()}}
OUT["survey"] = S

json.dump(OUT, open("results.json", "w"), indent=1, default=str)
print(json.dumps({k: OUT[k] for k in ["descriptive", "predictive", "prescriptive"]}, indent=1, default=str)[:6000])
print(json.dumps(OUT["diagnostic"]["cu_fail_count"]), OUT["diagnostic"]["corr_avgcw_avgmark"], OUT["diagnostic"]["exam_pct_of_max"], OUT["diagnostic"]["cw_pct_of_max"])
print(json.dumps(OUT["lms"]), json.dumps(OUT["survey"]))
