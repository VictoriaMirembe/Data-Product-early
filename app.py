"""
UICT Student Success Early-Warning — local dashboard
Run:  streamlit run app.py
Needs results.json and student_risk_scores.csv in this folder
(both are created by: python analysis.py)
"""
import json
import pandas as pd
import streamlit as st

st.set_page_config(page_title="UICT Student Success Early-Warning", layout="wide")

st.markdown("""
<style>
[data-testid="stApp"] {
    background: linear-gradient(135deg, #e0ecff 0%, #ece6ff 50%, #f6eefe 100%);
    background-attachment: fixed;
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] > div:first-child {
    background: rgba(255, 255, 255, 0.55);
    backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);
    border-right: 1px solid rgba(255, 255, 255, 0.6);
}
[data-testid="stMetric"],
[data-testid="stDataFrame"],
[data-testid="stVegaLiteChart"] {
    background: rgba(255, 255, 255, 0.55);
    backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.6);
    border-radius: 16px;
    box-shadow: 0 4px 24px rgba(60, 70, 120, 0.08);
    padding: 12px 16px;
}
[data-testid="stVegaLiteChart"] { padding: 0; }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load():
    R = json.load(open("results.json"))
    students = pd.read_csv("student_risk_scores.csv")
    return R, students

R, S = load()
D, G, P, X, L, SV = (R[k] for k in ["descriptive", "diagnostic", "predictive", "prescriptive", "lms", "survey"])
CUS = [f"CU{i}" for i in range(1, 7)]
rf = P["models"]["Random Forest"]

st.title("UICT Student Success Early-Warning")
st.caption("UICT diploma students · Semester 1, 2020 · Uganda Open Education Data Portal")
prog = {r["STUDENT_PROGRAM"]: r for r in D["by_program"]}
st.markdown(
    f"**{prog['DITB']['at_risk']*100:.0f}%** of DITB students ended the semester at risk versus "
    f"**{prog['DITS']['at_risk']*100:.0f}%** of DITS, and **CU6** produced "
    f"**{G['cu_fail_count'][5]} of {sum(G['cu_fail_count'])}** fails and retakes."
)
c = st.columns(5)
c[0].metric("Students analysed", D["students_complete"])
c[1].metric("Average GPA (/5)", f"{D['avg_cgpa']:.2f}")
c[2].metric("At risk", f"{D['at_risk_rate']}%")
c[3].metric("Model accuracy (test)", f"{rf['accuracy']:.0%}")
c[4].metric("At-risk caught (recall)", f"{rf['recall']:.0%}")

PAGES = ["1 · Descriptive", "2 · Diagnostic", "3 · Predictive", "4 · Prescriptive"]
with st.sidebar:
    st.header("Navigation")
    page = st.radio("Go to", PAGES, label_visibility="collapsed")
    st.divider()
    st.caption("Descriptive: what happened · Diagnostic: why · Predictive: who will be at risk · Prescriptive: what to do")
    st.caption("Team: Ssekajja Victoria Mirembe, Wagaba Vincent, Kwagala Rashida, Kyaligonza Gerald, Ekure Joseph")


if page == PAGES[0]:
    st.subheader("What happened?")
    st.dataframe(pd.DataFrame(D["by_program"]).rename(columns={
        "STUDENT_PROGRAM": "Programme", "n": "Students", "avg_mark": "Avg mark", "cgpa": "GPA", "at_risk": "At-risk share"}),
        hide_index=True)
    a, b = st.columns(2)
    with a:
        st.markdown("**Average mark per course unit, by programme**")
        st.bar_chart(pd.DataFrame(D["cu_avg"], index=CUS), stack=False)
    with b:
        st.markdown("**How all course-unit marks are spread**")
        st.bar_chart(pd.DataFrame({"Marks": D["mark_dist"]}, index=D["mark_dist_labels"]))
    a, b = st.columns(2)
    with a:
        st.markdown("**LMS: when learners last logged in**")
        st.bar_chart(pd.DataFrame({"Learners": L["recency"]}, index=L["recency_labels"]))
        st.caption(f"{L['never_pct']}% of learner accounts never logged in. Separate dataset, not linked to individual students.")
    with b:
        st.markdown("**Digital-skills trainees: what they expect**")
        st.bar_chart(pd.Series(SV["expectations"], name="Respondents").sort_values(ascending=False))
        st.caption(f"{SV['n']} valid responses · {SV['internet_yes_pct']}% have internet access")


if page == PAGES[1]:
    st.subheader("Why did it happen?")
    a, b = st.columns(2)
    with a:
        st.markdown("**Fails and retakes by course unit**")
        st.bar_chart(pd.DataFrame({"Fails + retakes": G["cu_fail_count"]}, index=CUS))
    with b:
        st.markdown("**Coursework vs exam, % of maximum**")
        cw = [v / 40 * 100 for v in D["cu_cw_ex"]["cw"]]
        ex = [v / 60 * 100 for v in D["cu_cw_ex"]["ex"]]
        st.bar_chart(pd.DataFrame({"Coursework %": cw, "Exam %": ex}, index=CUS), stack=False)
    a, b = st.columns(2)
    with a:
        st.markdown(f"**Coursework predicts the final mark** (correlation {G['corr_avgcw_avgmark']})")
        sc = S[["AVG_CW", "AVG_MARK"]].copy()
        sc["Status"] = S["AT_RISK"].map({1: "At risk", 0: "On track"})
        st.scatter_chart(sc, x="AVG_CW", y="AVG_MARK", color="Status")
    with b:
        st.markdown("**At-risk share by study mode**")
        sm = pd.DataFrame(D["by_study_time"]).set_index("STUDY_TIME")
        st.bar_chart(sm[["at_risk"]].rename(columns={"at_risk": "At-risk share"}))


if page == PAGES[2]:
    st.subheader("Who will be at risk?")
    st.write("Trained only on information available **before exams**: coursework in each unit, "
             f"programme, sex and study mode. {P['n_train']} students trained it, {P['n_test']} tested it.")
    st.dataframe(pd.DataFrame(P["models"]).T.drop(columns="confusion"), width="stretch")
    st.caption("CV = 5-fold cross-validation, the more honest estimate on a small sample. Recall matters most.")
    a, b = st.columns(2)
    with a:
        st.markdown("**What the model relies on (Random Forest)**")
        st.bar_chart(pd.DataFrame(P["importance"], columns=["Feature", "Importance"]).set_index("Feature"))
    with b:
        st.markdown("**Try it: predict a new student**")
        try:
            import joblib
            bundle = joblib.load("model.joblib")
            model, feats = bundle["model"], bundle["features"]
            row = {f"CU{i}_CW": st.slider(f"CU{i} coursework (/40)", 0, 40, 28) for i in range(1, 7)}
            programme = st.radio("Programme", ["DITB", "DITS"], horizontal=True)
            sex = st.radio("Sex", ["M", "F"], horizontal=True)
            mode = st.radio("Study mode", ["DAY", "EVENING", "WEEKEND"], horizontal=True)
            row.update({"IS_DITS": int(programme == "DITS"), "IS_FEMALE": int(sex == "F"),
                        "STUDY_DAY": int(mode == "DAY"), "STUDY_EVENING": int(mode == "EVENING"),
                        "STUDY_WEEKEND": int(mode == "WEEKEND")})
            p = model.predict_proba(pd.DataFrame([row])[feats])[0, 1]
            tier = "High" if p >= 0.7 else "Medium" if p >= 0.4 else "Low"
            st.metric("Predicted risk score", f"{p:.2f}", tier + " tier", delta_color="off")
        except Exception as e:
            st.info(f"Live prediction unavailable ({e}). Run python analysis.py first.")


if page == PAGES[3]:
    st.subheader("What should we do?")
    a, b = st.columns([1, 2])
    with a:
        tc = X["tier_counts"]
        for t in ["High", "Medium", "Low"]:
            st.metric(f"{t} tier", tc.get(t, 0))
    with b:
        st.markdown("""
- **Intervene on CU6 first**: review its exam and add revision clinics.
- **Mentor High-tier students** with weekly check-ins once coursework marks are in.
- **Prioritise DITB and evening classes**, which have the highest at-risk rates.
- **Coach exam technique**: students lose most marks in exams, not coursework.
- **Re-engage dormant LMS users**, a fifth of learner accounts never logged in.
""")
    st.markdown("**Student watchlist** (highest predicted risk first)")
    tiers = st.multiselect("Show tiers", ["High", "Medium", "Low"], default=["High", "Medium", "Low"])
    w = S.sort_values("RISK_PROB", ascending=False)
    w = w[w.TIER.isin(tiers)][["STUDENT_NO", "STUDENT_PROGRAM", "STUDY_TIME", "AVG_CW", "RISK_PROB", "TIER", "ACTION"]]
    st.dataframe(w.round(2), hide_index=True, width="stretch")
    st.download_button("Download watchlist (CSV)", w.to_csv(index=False), "watchlist.csv")
