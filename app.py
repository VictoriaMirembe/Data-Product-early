"""
UICT Student Success Early-Warning — local dashboard
Run:  streamlit run app.py
Needs results.json. Uses generated student_risk_scores.csv and model.joblib
when available; otherwise shows the summary and top-15 watchlist in results.json.
"""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
RESULTS_PATH = APP_DIR / "results.json"
SCORES_PATH = APP_DIR / "student_risk_scores.csv"
MODEL_PATH = APP_DIR / "model.joblib"
WATCHLIST_COLUMNS = [
    "STUDENT_NO", "STUDENT_PROGRAM", "STUDY_TIME", "AVG_CW",
    "RISK_PROB", "AT_RISK", "TIER", "ACTION",
]

st.set_page_config(
    page_title="UICT student success early warning",
    page_icon=":material/school:",
    layout="wide",
)


@st.cache_data
def load():
    with RESULTS_PATH.open(encoding="utf-8") as results_file:
        results = json.load(results_file)

    has_full_watchlist = SCORES_PATH.is_file()
    if has_full_watchlist:
        watchlist = pd.read_csv(SCORES_PATH)
    else:
        watchlist = pd.DataFrame(
            results["prescriptive"]["watchlist"], columns=WATCHLIST_COLUMNS
        )
    return results, watchlist, has_full_watchlist


R, S, has_full_watchlist = load()
D, G, P, X, L, SV = (
    R[k] for k in ["descriptive", "diagnostic", "predictive", "prescriptive", "lms", "survey"]
)
CUS = [f"CU{i}" for i in range(1, 7)]
rf = P["models"]["Random Forest"]
scatter_data = pd.DataFrame(
    G["scatter"],
    columns=["AVG_CW", "AVG_MARK", "AT_RISK", "STUDENT_PROGRAM"],
)
scatter_data["Status"] = scatter_data["AT_RISK"].map({1: "At risk", 0: "On track"})

st.title("Student success early warning", icon=":material/school:")
st.caption("UICT diploma students · Semester 1, 2020 · Uganda Open Education Data Portal")
if not has_full_watchlist:
    st.badge(
        "JSON-only view",
        icon=":material/info:",
        color="orange",
    )
    st.caption(
        "Precomputed results only · Top 15 watchlist entries · Live predictions unavailable"
    )
prog = {r["STUDENT_PROGRAM"]: r for r in D["by_program"]}
with st.container(border=True):
    st.markdown("**A clearer picture of student support needs**")
    st.markdown(
        f"{prog['DITB']['at_risk']*100:.0f}% of DITB students ended the semester at risk, "
        f"compared with {prog['DITS']['at_risk']*100:.0f}% of DITS students. "
        f"CU6 accounted for {G['cu_fail_count'][5]} of {sum(G['cu_fail_count'])} fails and retakes."
    )

with st.grid(5, min_column_width=150, gap="small"):
    st.metric("Students analysed", D["students_complete"], border=True, height="stretch")
    st.metric("Average GPA", f"{D['avg_cgpa']:.2f} / 5", border=True, height="stretch")
    st.metric("Students at risk", f"{D['at_risk_rate']}%", border=True, height="stretch")
    st.metric("RF test accuracy", f"{rf['accuracy']:.0%}", border=True, height="stretch")
    st.metric("RF test recall", f"{rf['recall']:.0%}", border=True, height="stretch")

PAGES = ["Descriptive", "Diagnostic", "Predictive", "Prescriptive"]
PAGE_ICONS = {
    "Descriptive": "insights",
    "Diagnostic": "troubleshoot",
    "Predictive": "query_stats",
    "Prescriptive": "checklist",
}
with st.sidebar:
    st.title("UICT", icon=":material/school:")
    st.caption("Student success · Semester 1, 2020")
    st.badge("Early warning dashboard", icon=":material/monitoring:", color="blue")
    page = st.radio(
        "Explore the analysis",
        PAGES,
        format_func=lambda name: f":material/{PAGE_ICONS[name]}:  {name}",
        captions=[
            "Cohort snapshot",
            "Patterns behind the results",
            "Model evaluation",
            "Recommended support",
        ],
        key="dashboard_page",
        width="stretch",
    )
    with st.container(border=True):
        st.markdown("**Data snapshot**")
        st.caption(f"{D['students_complete']} complete students · six course units")
        if not has_full_watchlist:
            st.caption("Summary mode: only saved results and the top 15 watchlist rows.")
    with st.expander("Project team", icon=":material/groups:"):
        st.caption(
            "Ssekajja Victoria Mirembe, Wagaba Vincent, Kwagala Rashida, "
            "Kyaligonza Gerald, Ekure Joseph"
        )


if page == PAGES[0]:
    st.subheader("What happened?", icon=":material/insights:")
    st.caption("A semester snapshot of performance, risk, and digital engagement.")
    with st.container(border=True):
        st.markdown("**Outcomes by programme**")
        st.dataframe(
            pd.DataFrame(D["by_program"]).rename(columns={
                "STUDENT_PROGRAM": "Programme",
                "n": "Students",
                "avg_mark": "Average mark",
                "cgpa": "GPA",
                "at_risk": "At-risk share",
            }),
            hide_index=True,
            alt="Student counts, average marks, GPA, and at-risk share by programme",
        )
    a, b = st.columns(2)
    with a:
        with st.container(border=True):
            st.markdown("**Average mark by course unit**")
            st.bar_chart(
                pd.DataFrame(D["cu_avg"], index=CUS),
                stack=False,
                alt="Average course-unit marks for DITB and DITS",
            )
    with b:
        with st.container(border=True):
            st.markdown("**Mark distribution**")
            st.bar_chart(
                pd.DataFrame({"Marks": D["mark_dist"]}, index=D["mark_dist_labels"]),
                alt="Distribution of course-unit marks across mark bands",
            )
    a, b = st.columns(2)
    with a:
        with st.container(border=True):
            st.markdown("**LMS login recency**")
            st.bar_chart(
                pd.DataFrame({"Learners": L["recency"]}, index=L["recency_labels"]),
                alt="Learner LMS accounts by last-login recency",
            )
            st.caption(
                f"{L['never_pct']}% of learner accounts never logged in. "
                "This separate dataset is not linked to individual students."
            )
    with b:
        with st.container(border=True):
            st.markdown("**Digital-skills trainee expectations**")
            st.bar_chart(
                pd.Series(SV["expectations"], name="Respondents").sort_values(ascending=False),
                alt="Digital-skills trainees grouped by expectation theme",
            )
            st.caption(
                f"{SV['n']} valid responses · {SV['internet_yes_pct']}% have internet access"
            )


if page == PAGES[1]:
    st.subheader("Why did it happen?", icon=":material/troubleshoot:")
    st.caption("Explore course-unit differences and the relationship between coursework and final marks.")
    with st.container(border=True):
        st.markdown("**Focus the student comparison**")
        programme_col, status_col = st.columns(2)
        with programme_col:
            selected_programmes = st.pills(
                "Programmes",
                ["DITB", "DITS"],
                default=["DITB", "DITS"],
                selection_mode="multi",
                key="scatter_programmes",
            )
        with status_col:
            selected_statuses = st.pills(
                "Risk status",
                ["At risk", "On track"],
                default=["At risk", "On track"],
                selection_mode="multi",
                key="scatter_statuses",
            )
    scatter_view = scatter_data[
        scatter_data["STUDENT_PROGRAM"].isin(selected_programmes or [])
        & scatter_data["Status"].isin(selected_statuses or [])
    ]
    a, b = st.columns(2)
    with a:
        with st.container(border=True):
            st.markdown("**Fails and retakes by course unit**")
            st.bar_chart(
                pd.DataFrame({"Fails + retakes": G["cu_fail_count"]}, index=CUS),
                alt="Fail and retake counts by course unit, highest for CU6",
            )
    with b:
        with st.container(border=True):
            st.markdown("**Coursework and exam performance**")
            cw = [v / 40 * 100 for v in D["cu_cw_ex"]["cw"]]
            ex = [v / 60 * 100 for v in D["cu_cw_ex"]["ex"]]
            st.bar_chart(
                pd.DataFrame({"Coursework %": cw, "Exam %": ex}, index=CUS),
                stack=False,
                alt="Coursework and exam marks as a percentage of their maximum by course unit",
            )
    a, b = st.columns(2)
    with a:
        with st.container(border=True):
            st.markdown(f"**Coursework and final mark** · correlation {G['corr_avgcw_avgmark']}")
            if scatter_view.empty:
                st.info("Choose at least one programme and one risk status to show students.")
            else:
                st.caption(f"Showing {len(scatter_view)} students")
                st.scatter_chart(
                    scatter_view,
                    x="AVG_CW",
                    y="AVG_MARK",
                    color="Status",
                    alt="Final marks generally rise with coursework marks; points are grouped by risk status",
                )
    with b:
        with st.container(border=True):
            st.markdown("**At-risk share by study mode**")
            sm = pd.DataFrame(D["by_study_time"]).set_index("STUDY_TIME")
            st.bar_chart(
                sm[["at_risk"]].rename(columns={"at_risk": "At-risk share"}),
                alt="At-risk share by study mode",
            )


if page == PAGES[2]:
    st.subheader("Who may need support?", icon=":material/query_stats:")
    st.caption("Recorded model results use coursework and other information available before exams.")
    st.write("Trained only on information available **before exams**: coursework in each unit, "
             f"programme, sex and study mode. {P['n_train']} students trained it, {P['n_test']} tested it.")
    with st.container(border=True):
        st.markdown("**Model performance**")
        st.dataframe(
            pd.DataFrame(P["models"]).T.drop(columns="confusion"),
            width="stretch",
            alt="Recorded test-set and cross-validation metrics for both models",
        )
        st.caption("Five-fold cross-validation is the more realistic estimate on this small sample.")
    a, b = st.columns(2)
    with a:
        with st.container(border=True):
            st.markdown("**What the Random Forest relied on**")
            st.bar_chart(
                pd.DataFrame(P["importance"], columns=["Feature", "Importance"]).set_index("Feature"),
                alt="Random Forest feature importance, led by CU6 coursework",
            )
    with b:
        with st.container(border=True):
            if MODEL_PATH.is_file():
                st.markdown("**Try a sample prediction**")
                import joblib

                bundle = joblib.load(MODEL_PATH)
                model, feats = bundle["model"], bundle["features"]
                row = {
                    f"CU{i}_CW": st.slider(f"CU{i} coursework (/40)", 0, 40, 28)
                    for i in range(1, 7)
                }
                programme = st.segmented_control(
                    "Programme", ["DITB", "DITS"], default="DITB"
                )
                sex = st.segmented_control("Sex", ["M", "F"], default="M")
                mode = st.segmented_control(
                    "Study mode", ["DAY", "EVENING", "WEEKEND"], default="DAY"
                )
                row.update({
                    "IS_DITS": int(programme == "DITS"),
                    "IS_FEMALE": int(sex == "F"),
                    "STUDY_DAY": int(mode == "DAY"),
                    "STUDY_EVENING": int(mode == "EVENING"),
                    "STUDY_WEEKEND": int(mode == "WEEKEND"),
                })
                probability = model.predict_proba(pd.DataFrame([row])[feats])[0, 1]
                tier = "High" if probability >= 0.7 else "Medium" if probability >= 0.4 else "Low"
                st.metric(
                    "Predicted risk score",
                    f"{probability:.2f}",
                    tier + " tier",
                    delta_color="off",
                    border=True,
                )
            else:
                st.markdown("**Live prediction**")
                st.info(
                    "Unavailable without model.joblib. The performance metrics shown "
                    "here are precomputed results."
                )


if page == PAGES[3]:
    st.subheader("What should we do?", icon=":material/checklist:")
    st.caption("Turn the strongest signals into practical, human-reviewed support.")
    tc = X["tier_counts"]
    with st.grid(3, min_column_width=150, gap="small"):
        for tier_name in ["High", "Medium", "Low"]:
            st.metric(
                f"{tier_name} support",
                tc.get(tier_name, 0),
                border=True,
                height="stretch",
            )
    with st.container(border=True):
        st.markdown("**Recommended priorities**")
        st.markdown("""
- Review **CU6** first: it accounts for most fails and retakes.
- Assign mentors and weekly check-ins to students in the **High** tier.
- Prioritise **DITB** and evening classes for additional review.
- Add exam-technique support; students lose a larger share of marks in exams.
- Re-engage inactive LMS users, while remembering that LMS records are not linked to these students.
""")
    with st.container(border=True):
        st.markdown("**Student watchlist**")
        st.caption("Highest predicted risk first.")
        if not has_full_watchlist:
            st.caption("Only the top 15 rows are available from results.json, not the full student list.")
        with st.container(horizontal=True, vertical_alignment="bottom"):
            tiers = st.pills(
                "Filter by support tier",
                ["High", "Medium", "Low"],
                default=["High", "Medium", "Low"],
                selection_mode="multi",
                key="watchlist_tiers",
            )
        w = S.sort_values("RISK_PROB", ascending=False)
        w = w[w.TIER.isin(tiers)][
            ["STUDENT_NO", "STUDENT_PROGRAM", "STUDY_TIME", "AVG_CW", "RISK_PROB", "TIER", "ACTION"]
        ]
        with st.container(horizontal=True, vertical_alignment="center"):
            st.caption(f"{len(w)} students shown")
            st.download_button(
                "Download CSV",
                w.to_csv(index=False),
                "available_watchlist.csv",
                icon=":material/download:",
            )
        st.dataframe(
            w.round(2),
            hide_index=True,
            width="stretch",
            alt="Available students ranked by predicted risk",
        )
