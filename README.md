# UICT Student Success Early-Warning

A data product that flags diploma students at risk of failing **before their exams**, using only their coursework marks, and recommends a level of support for each student.

It covers all four types of analytics on one semester of student results from the Uganda Open Education Data Portal:

| Type | Question | Output |
|---|---|---|
| Descriptive | What happened? | At-risk rates by programme, sex and study mode; marks per unit |
| Diagnostic | Why did it happen? | Which unit drives failure; coursework vs exam; correlation |
| Predictive | What will happen? | Random Forest model that gives each student a risk probability (`model.joblib`) |
| Prescriptive | What should we do? | High / Medium / Low support tier and an action per student (`student_risk_scores.csv`) |

## Datasets

| File | Contents | Role |
|---|---|---|
| `DITA student persformance 2020 sem1.csv` | 2020 Semester 1 results. Despite the file name, these are **DITB** students | Core |
| `DITS student performance 2020 sem1.csv` | 2020 Semester 1 results for DITS students | Core |
| `Learner management system usage.csv` | Last-login records from the institute's LMS | Context only |
| `students_survey_data digital skilling.csv` | Survey of digital-skilling participants (internet access, expectations) | Context only |

Each performance file has, per course unit (CU1–CU6): coursework (`CW`, /40), exam (`EX`, /60), final mark (`MK`, /100) and grade point (`GP`).

**Why the LMS and survey data are context only:** they cover different years and different populations, and they share **no student ID** with the performance files. There is no way to link a login record or a survey answer to a particular student. So they are summarised separately as background on digital engagement and are **not** used in the model.

## How to run

Requires Python 3.10+.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install pandas numpy scikit-learn matplotlib joblib jupyter

python analysis.py                      # full pipeline -> results.json, model.joblib, CSVs
jupyter notebook analysis_walkthrough.ipynb   # same pipeline, step by step with explanations
```

## How to run the dashboard

```bash
source .venv/bin/activate
pip install streamlit                 # only if it is not installed yet
python analysis.py                    # only if results.json, student_risk_scores.csv or model.joblib is missing
streamlit run app.py
```

Then open **http://localhost:8501** in a browser. The dashboard reads `results.json`, `student_risk_scores.csv` and `model.joblib`. Stop it with `Ctrl+C` in the terminal.

**Outputs**
- `clean_performance.csv`: cleaned data for the 81 complete students, with the `AT_RISK` label
- `student_risk_scores.csv`: every student's risk probability, tier and recommended action
- `model.joblib`: the trained Random Forest plus its feature list
- `results.json`: every number used in the analysis

## Key findings

- **Descriptive:** 30.9% of the 81 students are at risk. The gap between programmes is large: **51% of DITB** students against **14% of DITS** students.
- **Diagnostic:** **CU6 accounts for 16 of the 19** fail/retake cases. Marks are lost mostly in exams (62% of exam marks earned vs 74% of coursework marks).
- **Predictive:** a Random Forest using only coursework, programme, sex and study mode scored **0.90 accuracy / 0.83 recall** on the held-out test set. **5-fold cross-validated accuracy was about 0.70**, which is the more realistic estimate.
- **Prescriptive:** 18 students are placed in the High tier (mentor plus weekly check-ins), 14 in Medium (revision clinics) and 49 in Low. At institution level, **CU6 should be reviewed first.**

## Method notes

- **At risk** = failed any unit (< 50), **or** had a starred retake mark (e.g. `50*`), **or** CGPA < 3.0.
- **`50*` marks:** the number is kept and the star is recorded as a retake flag.
- **5 students** with marks in fewer than 6 units (likely dropouts or deferrals) are reported but excluded from the model.
- **Model inputs** are coursework marks only (no exam marks or grades), so the model can be used mid-semester, before exams. `random_state=42` throughout.

## Limitations

- **Small sample:** 81 students (only 21 in the test set), so the metrics are unstable. Test accuracy (0.90) and cross-validated accuracy (0.70) differ widely for this reason.
- **One semester, one year (2020 Sem 1):** there is no second cohort to check that the model works on future students.
- **Unnamed course units:** CU1–CU6 are not identified in the source data, so "review CU6" cannot yet be tied to a named course.
- **Weak on DITS:** with only 6 at-risk DITS students, the model leans heavily on programme and rarely flags at-risk DITS students in cross-validation.
- **No behavioural data:** attendance and LMS activity could not be linked to individual students.
- **Risk probabilities in the output file** come from the model refit on all 81 students, so they are more confident than they would be for new students.
