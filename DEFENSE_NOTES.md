# Defense notes: likely examiner questions

Numbers come from `analysis.py` / `results.json`.

---

### 1. "Our model shows 90% accuracy but also 70%. Which is it?"

"The honest figure is about **70%**. The 90% comes from a single test set of only **21 students**, and that particular group happened to be easy to predict. Cross-validation tests the model five times, so every one of the 81 students is used as test data exactly once, and the average is about 70%. The individual folds ranged from 50% to 82%, which shows how much a small sample moves the result. I report both, but I would use 70% when deciding whether to trust it."

### 2. "Why do we focus on recall rather than accuracy?"

"Recall is the share of truly at-risk students the model catches. In an early-warning system the expensive mistake is **missing** a struggling student, because then no help reaches them. A false alarm only means an extra mentoring session for a student who was fine, which costs little. Accuracy can also mislead: about 69% of students are not at risk, so a model that flags nobody would still be 69% accurate and catch no one. On the test set the Random Forest caught **5 of the 6** at-risk students (recall 0.83) with one false alarm."

### 3. "Then why not pick Logistic Regression? It had recall 1.0."

"On the test set it caught all 6, but it also raised **4 false alarms** out of 10 flags. In cross-validation its recall (0.68) was actually higher than the Random Forest's (0.56), and its accuracy (0.67) was slightly lower. So the two models are close, and neither is clearly better on 81 students. I used the Random Forest for the risk scores because it scored better on test accuracy and precision and gives clear feature importances. Choosing Logistic Regression for maximum recall would also be defensible, and that is a trade-off I would revisit with more data."

### 4. "How did we handle marks like `50*`?"

"The star marks a retake or supplementary result. The data portal does not document it, so that is my interpretation. I kept the number, so 50* counts as 50 in averages, and I recorded the star in a separate retake flag. I did that because deleting the star would hide the fact that the student needed a second attempt, which is a strong risk signal. There are 14 starred marks across 13 students, and 12 of them are in CU6. Eight are exactly 50, which looks like marks capped at the pass mark after a retake. Any student with a starred mark counts as at risk."

### 5. "Why did we exclude 5 students?"

"There were 86 student records, and 5 of them had final marks in only 1 to 4 of the 6 units. That pattern usually means a student dropped out, deferred or missed exams. I could not judge their performance fairly, and filling in the missing marks would mean inventing data. So I report them separately, since they are a risk group in their own right, but leave them out of the model. That leaves 81 complete students."

### 6. "Is the model biased by sex or study mode?"

"I checked two things. First, sex and study mode together make up **under 7% of the feature importance**. The model is driven by coursework marks and programme. Second, I compared cross-validated predictions by group. The model flags each sex at roughly its real at-risk rate: 26% flagged vs 21% truly at risk for women, and 35% vs 34% for men. Evening students are flagged more often than they are truly at risk (56% vs 44%), so it over-flags that group a little. But there are only 19 women and 16 evening students, too few to prove or rule out bias. Dropping sex and study mode entirely gives the same 0.70 accuracy, so they could be removed for fairness at no cost. **The bigger issue is programme:** in cross-validation the model caught **none of the 6 at-risk DITS students**. It has mainly learned 'DITB = risky', so DITS students need a human check too."

### 7. "Why only coursework as model inputs and not exam marks?"

"Because the point is to predict **before exams**, when there is still time to help. The at-risk label is built from final marks, and final marks include the exam. If I used exam marks, the model would be reading the answer: accuracy would be close to 100% but useless in practice. This mistake is called data leakage. Coursework is available mid-semester, so the model can run at that point."

### 8. "How did we define 'at risk', and isn't it arbitrary?"

"A student is at risk if they failed any unit, had a retake, or have a CGPA below 3.0, the pass-class boundary. 25 of 81 students qualify. I included the CGPA condition because outright fails are rare, only 6 of 486 unit marks, so counting fails alone would miss students who are only just passing. 8 students are flagged on CGPA alone. It is a judgement call. The rule is written down in one line of code, so an institution could change it and rerun the pipeline."

### 9. "What does our prescriptive part actually recommend, and why should anyone trust it?"

"Each student gets a tier from their risk probability. **High** (≥ 0.7) means a mentor and weekly check-ins, **Medium** (0.4 to 0.7) means revision clinics, and **Low** means standard support. Any unit where their coursework is below 25/40 is named as a remedial focus. That gives 18 High, 14 Medium and 49 Low. These are simple, readable rules rather than another model, so staff can see why a student is in a tier and adjust the thresholds. At institution level the clearest recommendation is to **review CU6**, which accounts for 16 of the 19 fail/retake cases. Honestly, the thresholds are sensible defaults, not optimised, and the probabilities come from a model trained on these same students, so they look more certain than they would for a new cohort."

### 10. "What are the main limitations, and what would we do next?"

"Three main ones. First, **81 students in one semester** is a small sample, which is why the accuracy figures are unstable. Second, there is **no second cohort** to show the model works on future students. Third, the **course units are unnamed**, so I cannot say which real course CU6 is. The LMS and survey data could not be linked to students, so they are context only. Next steps would be to collect more semesters, test on a later cohort, get the unit names and a student-linked attendance or LMS signal, and retrain the model with those."

---

### Quick numbers to remember

| | |
|---|---|
| Records / complete / excluded | 86 / 81 / 5 |
| At risk overall | 30.9% (25 of 81) |
| DITB vs DITS at risk | 51% (19/37) vs 14% (6/44) |
| CU6 fails/retakes | 16 of 19 |
| Coursework vs final mark correlation | r = 0.66 |
| Exam vs coursework (% of max) | 62% vs 74% |
| Random Forest test: accuracy / recall / precision | 0.90 / 0.83 / 0.83 |
| Random Forest 5-fold CV: accuracy / recall | 0.70 / 0.56 |
| Logistic Regression test: accuracy / recall / precision | 0.81 / 1.00 / 0.60 |
| Train / test split | 60 / 21 (stratified, random_state=42) |
| Tiers High / Medium / Low | 18 / 14 / 49 |
| Top feature | CU6 coursework (0.223) |
