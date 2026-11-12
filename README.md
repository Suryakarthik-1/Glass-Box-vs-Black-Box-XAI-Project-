# Glass Box vs Black Box
### Explainable diabetes risk prediction — XAI Mini Project (Modules 1 & 2)

> **One-line summary:** We trained one black-box model and three interpretable ("glass-box") models to predict diabetes, built an interactive web demo that shows how each model explains its decision, and measured the accuracy, complexity and fidelity of each. The interpretable GAM was **as accurate as the black box (AUC 0.826 vs 0.814)** while being fully readable.

---

## Table of contents

1. [Problem statement](#1-problem-statement)
2. [Objectives](#2-objectives)
3. [How the project maps to the syllabus](#3-how-the-project-maps-to-the-syllabus)
4. [Dataset](#4-dataset)
5. [Data preprocessing](#5-data-preprocessing)
6. [Models and how each one explains itself](#6-models-and-how-each-one-explains-itself)
7. [Evaluating interpretability](#7-evaluating-interpretability)
8. [Results](#8-results)
9. [The interactive demo](#9-the-interactive-demo)
10. [Presentation script (step by step)](#10-presentation-script-step-by-step)
11. [How to run the code](#11-how-to-run-the-code)
12. [Project structure](#12-project-structure)
13. [Limitations and future work](#13-limitations-and-future-work)
14. [Viva / Q&A preparation](#14-viva--qa-preparation)
15. [Key terms glossary](#15-key-terms-glossary)
16. [References](#16-references)

---

## 1. Problem statement

Machine learning models are increasingly used in healthcare to predict diseases. The most accurate models, such as gradient boosting, random forests and neural networks, are **black boxes**: they give a prediction (for example, "72% chance of diabetes") but cannot say *why*.

In medicine this is a serious problem:

- A doctor cannot act on a number without a reason.
- A patient has the right to know why a decision was made about them.
- A model may have learned a wrong or biased pattern, and nobody would notice.

**Question this project answers:**
> *Do we really need a black box for medical prediction, or can an interpretable model be just as accurate while also explaining itself?*

---

## 2. Objectives

1. Build a **black-box model** (Gradient Boosting) and show why it is hard to explain.
2. Build three **interpretable (ante-hoc) models** from Module 2:
   - Logistic Regression
   - Decision Tree
   - Generalized Additive Model (GAM)
3. Show **global** explanations (how the model works overall) and **local** explanations (why it made a decision for one patient) for each model.
4. **Evaluate interpretability** using accuracy, **complexity** and **fidelity**.
5. Present everything in an **interactive web demo** in which a patient's values can be changed live.

---

## 3. How the project maps to the syllabus

| Syllabus topic | Where it appears in the project |
|---|---|
| **M1:** The "black box" problem | *Black box* tab: gradient boosting gives a risk with no reason |
| **M1:** Need for explainability: transparency, trust, fairness, safety | *Black box* tab: table showing what goes wrong without explanation |
| **M1:** Principles of XAI | Every model shows *what* it predicts and *why* |
| **M1:** Global vs local explanations | Every tab is tagged **Global** / **Local**; each model shows both |
| **M1:** Ante-hoc vs post-hoc | Logistic regression, tree and GAM are **ante-hoc**; permutation importance and the surrogate tree are **post-hoc** |
| **M2:** Linear and logistic regression (interpreting coefficients) | *Logistic regression* tab: equation, weights, odds ratios, per-patient contributions |
| **M2:** Decision trees and rule-based models | *Decision tree* tab: tree diagram, highlighted path, IF–THEN rule list |
| **M2:** Generalized Additive Models | *GAM* tab: one shape curve per feature |
| **M2:** Evaluating interpretability (complexity, fidelity) | *Evaluate* tab: results table and fidelity chart |

---

## 4. Dataset

**Pima Indians Diabetes Database**, originally from the National Institute of Diabetes and Digestive and Kidney Diseases (USA). It is a standard benchmark dataset for medical machine learning.

| Property | Value |
|---|---|
| Patients | 768 (all female, age 21+, Pima Indian heritage) |
| Target | `Outcome`: 1 = has diabetes, 0 = no diabetes |
| Class balance | 500 non-diabetic (65%), 268 diabetic (35%) |
| Train / test split | 576 train / 192 test (75% / 25%, stratified, `random_state=42`) |

### Features used (6)

| Feature | Meaning | Unit |
|---|---|---|
| Pregnancies | Number of times pregnant | count |
| Glucose | Plasma glucose 2 hours after an oral glucose tolerance test | mg/dL |
| BloodPressure | Diastolic blood pressure | mm Hg |
| BMI | Body Mass Index | kg/m² |
| DiabetesPedigree | Diabetes Pedigree Function, a score of family history of diabetes (shown as **"Family history score"** in the demo) | score |
| Age | Age | years |

### Features dropped (2)
- **SkinThickness** and **Insulin** were dropped because a large share of their values are 0, which means "not measured". Filling so many gaps would add noise, and fewer features make the explanations easier to read.

---

## 5. Data preprocessing

1. **Missing values:** a value of `0` for Glucose (5 rows), BloodPressure (35 rows) and BMI (11 rows) is medically impossible, so it means "not recorded". These were replaced with the **median of the training set**:
   - Glucose → 117 mg/dL
   - BloodPressure → 72 mm Hg
   - BMI → 32.4 kg/m²

   Medians are computed **only on the training data** to avoid data leakage.
2. **Stratified split** keeps the 35% diabetic ratio the same in train and test.
3. **Standardisation** (z-score) is applied **only for logistic regression**, so its coefficients can be compared with each other. Trees, the GAM and the black box use raw values.

---

## 6. Models and how each one explains itself

### 6.1 Black box: Gradient Boosting Classifier

**What it is:** an ensemble of **150 decision trees** (depth 3, learning rate 0.05). Each new tree is trained to correct the mistakes of the trees before it. The final prediction is:

```
score = initial value + 0.05 × (tree₁ + tree₂ + … + tree₁₅₀)
P(diabetes) = sigmoid(score)
```

**Why it is a black box:** the model has **2,184 decision nodes**. No human can follow 150 trees for one patient, and no single rule or number explains the result.

**Post-hoc global explanation: permutation importance.** We shuffle one feature's values and measure how much the test ROC-AUC drops. A big drop means the model relies on that feature.

| Feature | Drop in AUC when shuffled |
|---|---|
| Glucose | 0.137 |
| BMI | 0.064 |
| Age | 0.015 |
| Pregnancies | 0.008 |
| Family history | 0.004 |
| Blood pressure | ≈ 0 |

**Limitation (the key point):** this tells us *what* the model uses overall, but not *how* (does higher glucose raise or lower the risk?) and nothing about *this particular patient*. That is the black-box problem.

---

### 6.2 Logistic Regression (ante-hoc)

**What it is:** a linear model in log-odds.

```
log-odds = b₀ + w₁·z(Pregnancies) + w₂·z(Glucose) + … + w₆·z(Age)
P(diabetes) = 1 / (1 + e^(−log-odds))
```

where `z(x) = (x − mean) / standard deviation`.

**Learned model:**

| Feature | 1 SD (unit) | Weight w | Odds ratio eʷ | Meaning |
|---|---|---|---|---|
| Glucose | 30.1 mg/dL | **+1.15** | **×3.14** | +30 mg/dL glucose multiplies the odds of diabetes by about 3 |
| BMI | 6.74 kg/m² | +0.69 | ×2.00 | +6.7 BMI doubles the odds |
| Pregnancies | 3.3 | +0.44 | ×1.55 | +3 pregnancies raises the odds by 55% |
| Family history | 0.33 | +0.21 | ×1.23 | |
| Age | 11.9 years | +0.12 | ×1.13 | |
| Blood pressure | 12.3 mm Hg | −0.06 | ×0.95 | Almost no effect |
| Bias b₀ | | −0.87 | | Baseline (an average patient has ≈30% risk) |

**How to read a coefficient:**
- **Sign:** + raises the risk, − lowers it.
- **Size:** because features are standardised, a bigger absolute weight means a more important feature.
- **Odds ratio:** the factor by which the odds change for a +1 SD increase, with all other features held fixed.

**Global explanation:** the weight table above.
**Local explanation:** for one patient, each feature's contribution is `wᵢ × zᵢ`. The demo shows these as red bars (raise risk) and blue bars (lower risk). They add up exactly to the log-odds, so the explanation is **complete and exact**.

**Limitation:** it assumes every effect is a straight line in log-odds, so it cannot capture patterns like "risk rises with age, then falls".

---

### 6.3 Decision Tree (ante-hoc)

**What it is:** a flowchart of yes/no questions. Settings: `max_depth = 3`, `min_samples_leaf = 20`, which gives **8 leaves = 8 rules**.

**The learned tree:**

```
Glucose ≤ 132.5 ?
├── yes → Age ≤ 30.5 ?
│        ├── yes → BMI ≤ 39.9 ?   → yes: 9% risk (n=221)  | no: 35% risk (n=20)
│        └── no  → BMI ≤ 26.9 ?   → yes: 0% risk (n=29)   | no: 43% risk (n=127)
└── no  → Glucose ≤ 154.5 ?
         ├── yes → BMI ≤ 29.95 ?  → yes: 25% risk (n=28)  | no: 60% risk (n=58)
         └── no  → Pregnancies ≤ 6.5 ? → yes: 78% risk (n=63) | no: 97% risk (n=30)
```

(`n` = number of training patients that land in that leaf; risk = share of them who were diabetic.)

**Global explanation:** the whole tree, or the same tree written as **IF–THEN rules**, for example:

```
IF Glucose > 132.5 AND Glucose ≤ 154.5 AND BMI > 29.95 THEN risk 60%
```

**Local explanation:** the **path** a patient takes from the root to a leaf. The demo highlights it, and the active rule is highlighted in the rule list.

**What the tree learned:** glucose is the first and most important split. Both glucose and BMI appear in several places, which matches the logistic regression and the black box.

**Limitation:** it is the least accurate model (AUC 0.760). A depth-3 tree gives only 8 possible risk values, so its predictions are coarse. Making it deeper improves accuracy but makes it harder to read. That is the **interpretability vs accuracy trade-off**.

---

### 6.4 Generalized Additive Model, GAM (ante-hoc)

**What it is:** a GAM keeps the *additive* structure of logistic regression but replaces each straight-line weight with a **smooth curve** fᵢ:

```
log-odds = b₀ + f₁(Pregnancies) + f₂(Glucose) + f₃(BloodPressure) + f₄(BMI) + f₅(FamilyHistory) + f₆(Age)
```

**How it was built:** each feature is expanded into **cubic B-spline basis functions** (`SplineTransformer`, 5 knots, degree 3). A regularised logistic regression (`C = 0.5`) then learns the weights for these basis functions. Adding up a feature's weighted splines gives its shape curve fᵢ(x). Each curve is centred so that its average over the training data is 0, and b₀ = −0.82.

**What the curves show (global explanation):**

| Feature | Shape of the curve | Range of effect (log-odds) |
|---|---|---|
| Glucose | Rises steeply and almost continuously. **Strongest feature.** | −1.02 to +1.71 |
| BMI | Rises up to about 50 kg/m², then flattens or dips (few patients there) | −0.97 to +0.79 |
| Age | Rises until about **45 years**, then **falls** | −0.44 to +0.57 |
| Pregnancies | Rises up to about 10, then flattens | −0.20 to +0.52 |
| Family history | Rises up to about 1.0, then flattens | −0.36 to +0.42 |
| Blood pressure | Almost flat, so **no real effect** | −0.13 to +0.05 |

The **Age** curve is something logistic regression cannot show. Logistic regression says "older means higher risk" with a single small weight. The GAM shows that risk peaks in middle age, possibly because this dataset contains few older diabetic patients.

**Local explanation:** read each curve at the patient's value (the dot in the demo) and add the values up:

```
log-odds = −0.82 − 0.07 + 0.46 + 0.03 + 0.25 − 0.17 + 0.28 = −0.04  →  P = 49%
```

**Why GAMs are powerful:** they keep the **"each feature separately"** readability of linear models while capturing **non-linear** effects. That is why the GAM was the most accurate model in this project.

**Limitation:** a basic GAM ignores **interactions** (for example, "high glucose is worse when BMI is also high"). Models such as GA²M / Explainable Boosting Machines add selected pairwise interactions.

---

## 7. Evaluating interpretability

Module 2 asks how to *measure* interpretability. We use three measures.

### 7.1 Accuracy (is the model correct?)
- **Accuracy:** % of test patients classified correctly at a 50% threshold.
- **ROC-AUC:** how well the model ranks diabetic patients above non-diabetic ones, regardless of threshold (0.5 = random, 1.0 = perfect). AUC is the fairer measure here because the classes are imbalanced (35% / 65%).

### 7.2 Complexity (is the explanation simple enough?)
How much a human must read to understand the whole model:

| Model | Complexity |
|---|---|
| Black box | 150 trees, 2,184 nodes |
| Logistic regression | 7 numbers |
| Decision tree | 8 rules, depth 3 |
| GAM | 6 curves |

### 7.3 Fidelity (is the explanation faithful to the model?)
**Fidelity** = how often an explanation model gives the **same answer as the black box** it is explaining.

**Experiment (post-hoc global surrogate):**
1. Take the black box's predictions on the training data as new labels.
2. Train decision trees of depth 1 to 8 to **copy the black box**.
3. On the test set, measure:
   - **Fidelity:** agreement with the black box's predictions.
   - **Accuracy:** agreement with the real outcome.

| Depth | Rules (leaves) | Fidelity | Accuracy |
|---|---|---|---|
| 1 | 2 | 81.8% | 68.2% |
| 2 | 4 | 83.9% | 72.4% |
| **3** | **8** | **89.1%** | 70.3% |
| 4 | 15 | 88.5% | 70.8% |
| 5 | 25 | 87.5% | 74.0% |
| 6 | 34 | 88.0% | 72.4% |
| 7 | 44 | 88.0% | 71.4% |
| 8 | 48 | 88.0% | 71.4% |

**Interpretation:** an 8-rule tree reproduces the black box's decision for **89% of patients**. Adding more rules (more complexity) **does not improve fidelity**. This is the **complexity vs fidelity trade-off**, and it shows there is a "sweet spot" where an explanation is both simple and faithful.

---

## 8. Results

| Model | Accuracy | ROC-AUC | Complexity | Explanation type | Scope |
|---|---|---|---|---|---|
| Black box (gradient boosting) | 72.9% | 0.814 | 150 trees · 2,184 nodes | Post-hoc only | Global (permutation) |
| Logistic regression | 71.9% | 0.823 | 7 numbers | Ante-hoc | Global + local |
| Decision tree (depth 3) | 71.4% | 0.760 | 8 rules | Ante-hoc | Global + local |
| **GAM (splines)** | **74.0%** | **0.826** | **6 curves** | **Ante-hoc** | **Global + local** |

### Key findings
1. **The GAM was the best model overall:** highest accuracy (74.0%) and highest AUC (0.826), beating the black box (0.814).
2. **Logistic regression also beat the black box on AUC** (0.823 vs 0.814) with only 7 numbers.
3. **All models agree on what matters:** glucose is the strongest factor, then BMI; blood pressure has almost no effect. Agreement across very different models makes the explanation more trustworthy.
4. **The decision tree is the easiest to read but the least accurate**, which shows the interpretability vs accuracy trade-off.
5. **Conclusion:** for structured (tabular) medical data, *interpretability does not have to cost accuracy*. This supports Cynthia Rudin's argument (2019) that for high-stakes decisions we should **use interpretable models instead of explaining black boxes**.

### The "Models disagree" patient (important for the demo)
A real test-set patient: Pregnancies 5, Glucose 136, BP 84, BMI 35.0, Family history 0.29, Age 35. **Actual outcome: has diabetes.**

| Model | Prediction | Correct? |
|---|---|---|
| Black box | 35% (low risk) | ✗ Wrong |
| Logistic regression | 48% | ✗ Just below 50% |
| GAM | 49% | ✗ Just below 50% |
| Decision tree | **60% (high risk)** | ✓ Correct |

The black box was wrong **and could not say why**. The decision tree was right and shows its exact reason: *Glucose > 132.5 AND Glucose ≤ 154.5 AND BMI > 29.95 → 60% risk*. The GAM and logistic regression show the patient is borderline, and which features push the risk up (glucose, BMI, age) and down (family history).

---

## 9. The interactive demo

The demo is a single web page that runs entirely in the browser. The models were trained in Python and their parameters were exported, so the page computes predictions live. The browser predictions were checked against Python and match to 4 decimal places.

### Layout
- **Left panel, Patient:** 6 sliders and 3 preset buttons (*High risk*, *Low risk*, *Models disagree*). Presets are real patients from the test set, with their true outcome shown.
- **Left panel, Predicted diabetes risk:** all four models' predictions side by side, plus a message saying whether they agree.
- **Right panel, 5 tabs:**

| Tab | What to show |
|---|---|
| **Black box** | The risk number with "Why? The model cannot say". Permutation importance chart. Table of why explainability matters (transparency, trust, fairness, safety). |
| **Logistic regression** | Full equation with numbers. Bar chart of this patient's contributions. Table of weights and odds ratios. |
| **Decision tree** | Tree diagram with this patient's path highlighted. List of all 8 rules with the active one highlighted. |
| **GAM** | 6 shape curves on a shared scale, a dot showing this patient's position on each, and the sum equation. |
| **Evaluate** | Results table (accuracy, AUC, complexity, ante-hoc/post-hoc, global/local). Fidelity chart vs tree depth. |

---

## 10. Presentation script (step by step)

**Total time: about 10–12 minutes.** Tab names are in **bold**.

### Slide/intro (1 min)
> "AI models in healthcare are often black boxes. They give a prediction but no reason. Our project asks: do we really need a black box, or can an interpretable model be as accurate and explain itself? We used the Pima Indians Diabetes dataset of 768 patients and compared one black-box model with three interpretable models from Module 2."

### Step 1: The black-box problem (2 min)
1. Click **"Models disagree"** in the left panel.
2. Stay on the **Black box** tab.
> "The gradient boosting model says this patient has a 35% risk. Why? It cannot tell us. It is 150 trees with 2,184 decision nodes."
3. Point to the permutation importance chart.
> "A post-hoc method tells us the model relies mostly on glucose overall. But it doesn't tell us *how* or anything about *this* patient."
4. Point to the risk panel: *"Models disagree: 1 of 4 say high risk."*
> "This patient actually has diabetes. The black box got it wrong, and we would never know why."

### Step 2: Logistic regression (2 min)
1. Click the **Logistic regression** tab.
> "This model is one equation. Each feature gets a weight."
2. Point to the table:
> "Glucose has the biggest weight, 1.15. Its odds ratio is 3.1, so every 30 mg/dL increase in glucose triples the odds of diabetes."
3. Point to the bar chart:
> "For this patient, glucose and BMI push the risk up (red). The baseline and family history pull it down (blue). The bars add up exactly to the prediction, 48%. That is a local explanation."
4. **Live demo:** drag the **Glucose** slider up to about 170 and watch the red bar grow and the risk rise.

### Step 3: Decision tree (2 min)
1. Click **Models disagree** again to reset, then click the **Decision tree** tab.
> "A decision tree is a flowchart. This patient's path is highlighted: glucose above 132.5, but not above 154.5, and BMI above 29.95. That puts them in a leaf where 60% of similar training patients had diabetes."
2. Point to the rule list:
> "The whole model is just 8 IF–THEN rules: a global explanation. The highlighted rule is the local explanation. The tree got this patient right, and we can see exactly why."
3. **Live demo:** drag **BMI** below 29.95 and the path jumps to a 25% leaf.

### Step 4: GAM (2 min)
1. Click the **GAM** tab.
> "A GAM keeps the 'one feature at a time' idea but uses a smooth curve instead of a straight line."
2. Point to the curves:
> "Glucose rises steeply, the strongest effect. Blood pressure is almost flat, no effect. And age is interesting: risk rises until about 45, then falls. Logistic regression can't show that, because it only fits straight lines."
3. Point to the equation:
> "The dots on each curve add up to the prediction: 49%."

### Step 5: Evaluation (2 min)
1. Click the **Evaluate** tab.
> "Now the key question: did we lose accuracy by being interpretable?"
2. Point to the table:
> "No. The GAM has the best AUC, 0.826, higher than the black box's 0.814, with just 6 curves instead of 2,184 nodes. Logistic regression also beats the black box."
3. Point to the fidelity chart:
> "We also measured fidelity. We trained small trees to copy the black box. An 8-rule tree agrees with the black box 89% of the time, and adding more rules doesn't help. That's the complexity vs fidelity trade-off from Module 2."

### Conclusion (1 min)
> "Three takeaways. One: black boxes give no reasons, which is dangerous in healthcare. Two: interpretable models give both global and local explanations. Three: on this medical data, the interpretable GAM was as accurate as the black box. For high-stakes decisions, we should prefer models that explain themselves."

### Backup demo ideas if asked
- Click **Low risk**: all four models agree (all low), and every explanation points to low glucose (57 mg/dL).
- Click **High risk**: all four agree on high risk; glucose of 184 dominates every explanation.
- Set Blood pressure to its minimum and then its maximum: almost nothing changes in any model, which shows it is an unimportant feature.

---

## 11. How to run the code

### Requirements
- Python 3.9+
- `numpy`, `pandas`, `scikit-learn` (version 1.1 or newer for `SplineTransformer`)

```bash
pip install numpy pandas scikit-learn
```

### Run
```bash
python train_models.py
```

### What it does
1. Loads `diabetes.csv`, cleans it and splits it into train and test sets.
2. Trains all four models.
3. Computes permutation importance, the surrogate fidelity curve and the test metrics.
4. Prints the decision tree as text, the results table and the fidelity table.
5. Writes `models.json` (all model parameters, used by the web demo) and `reference_predictions.json` (used to check that the web demo matches Python).

### Expected output (summary)
```
gbm     acc=0.729 auc=0.814  150 trees · 2184 nodes
logreg  acc=0.719 auc=0.823  7 numbers (6 weights + 1 bias)
tree    acc=0.714 auc=0.760  8 rules, depth 3
gam     acc=0.740 auc=0.826  6 shape curves
```

---

## 12. Project structure

```
xai_mini_project/
├── README.md                    ← this file
├── train_models.py              ← trains all models, evaluates, exports models.json
├── diabetes.csv                 ← Pima Indians Diabetes dataset (768 rows, no header)
├── models.json                  ← (generated) model parameters for the web demo
├── reference_predictions.json   ← (generated) Python predictions for checking the demo
└── Web demo                     ← "Glass Box vs Black Box" page (HTML + JavaScript)
```

### Code walkthrough of `train_models.py`

| Section | What it does |
|---|---|
| `data` | Reads the CSV, keeps 6 features, stratified 75/25 split, fills zero values with training medians |
| `black box` | `GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.05)`; exports every tree; computes permutation importance (20 repeats, ROC-AUC) |
| `logistic regression` | `StandardScaler` + `LogisticRegression(C=1.0)`; exports means, scales, weights, bias |
| `decision tree` | `DecisionTreeClassifier(max_depth=3, min_samples_leaf=20)`; prints rules with `export_text` |
| `GAM` | One `SplineTransformer(n_knots=5, degree=3)` per feature + `LogisticRegression(C=0.5)`; evaluates each shape curve on 80 points and centres it |
| `fidelity` | Trains surrogate trees of depth 1–8 on the black box's labels; measures fidelity and accuracy |
| `examples` | Picks 3 real test patients: high risk, low risk, and one where the black box and the tree disagree |
| `export` | Writes everything to `models.json` |

### How the web demo computes predictions (same maths as Python)
- **Black box:** walks each of the 150 trees to a leaf, adds `0.05 × leaf value` to the initial score, then applies the sigmoid.
- **Logistic regression:** `sigmoid(b + Σ wᵢ·(xᵢ − meanᵢ)/sdᵢ)`.
- **Decision tree:** walks the tree; the leaf's diabetic fraction is the risk.
- **GAM:** linear interpolation on each 80-point shape curve, summed with the intercept, then the sigmoid.

---

## 13. Limitations and future work

### Limitations
- **Small dataset:** 768 patients, 192 in the test set. Results could change with a different split; cross-validation would give more reliable numbers.
- **Narrow population:** only female patients of Pima Indian heritage, aged 21+. The models may not generalise to other groups, which is itself a **fairness** concern.
- **Missing data:** zeros were filled with medians, and two features were dropped.
- **No interactions in the GAM:** effects such as "glucose matters more when BMI is high" are not captured.
- **Permutation importance can mislead** when features are correlated (e.g. Age and Pregnancies).
- **Not a medical tool:** for teaching only.

### Future work
- Add **post-hoc local methods** (LIME, SHAP from Module 3) to explain the black box per patient, and compare them with the glass-box explanations.
- Use **Explainable Boosting Machines (GA²M)** to add pairwise interactions to the GAM.
- Use **k-fold cross-validation** and confidence intervals.
- Add **counterfactual explanations**: "what is the smallest change that would lower this patient's risk below 50%?"
- Run a **user study** with doctors or students to measure how understandable each explanation is.
- Test **fairness** across age groups.

---

## 14. Viva / Q&A preparation

**Q1. What is the difference between interpretability and explainability?**
Interpretability is how well a human can understand the model *itself*, such as a short decision tree. Explainability is the ability to give reasons for a model's output, which can be done even for black boxes using post-hoc methods.

**Q2. What is the difference between ante-hoc and post-hoc explanations?**
Ante-hoc (intrinsic) models are interpretable by design: logistic regression, decision trees, GAMs. Post-hoc methods explain a model *after* it is trained, treating it as a black box: permutation importance, surrogate models, LIME, SHAP.

**Q3. Global vs local explanation?**
Global explains how the model behaves overall (e.g. the weight table, the full rule set, the GAM curves). Local explains one specific prediction (e.g. this patient's contribution bars, highlighted path, dots on the curves).

**Q4. Why did you standardise features for logistic regression?**
So that coefficients are on the same scale and can be compared. Without it, a weight for glucose (in mg/dL) and a weight for family history (a 0–2.4 score) could not be compared directly.

**Q5. What is an odds ratio?**
`e^weight`. It is the factor by which the odds of the outcome multiply for a one-unit increase in the feature (here one standard deviation). Glucose's odds ratio of 3.14 means +1 SD (30 mg/dL) roughly triples the odds of diabetes.

**Q6. Why is the tree limited to depth 3?**
For interpretability. A depth-3 tree has at most 8 rules, which a person can read. Deeper trees become as hard to follow as a black box.

**Q7. What is a GAM and how is it different from logistic regression?**
Both are additive: each feature contributes separately. Logistic regression uses a straight line per feature (one weight). A GAM uses a flexible smooth curve per feature, so it can capture non-linear effects (like age peaking at 45) while staying interpretable.

**Q8. How did you build the GAM?**
Each feature was expanded into cubic B-spline basis functions (5 knots, degree 3) using scikit-learn's `SplineTransformer`. A regularised logistic regression was then fitted on all the basis functions together. The weighted sum of a feature's basis functions is its shape curve.

**Q9. What is fidelity?**
The degree to which an explanation (here a surrogate tree) agrees with the model it explains. We measured it as the percentage of test patients for whom the surrogate tree gives the same class as the black box. The best was 89.1% at depth 3.

**Q10. What is the difference between fidelity and accuracy?**
Fidelity compares the explanation with the **black box's predictions**. Accuracy compares it with the **true labels**. A surrogate can be very faithful to a black box that is itself wrong.

**Q11. What is permutation importance?**
A post-hoc, model-agnostic, global method. You shuffle one feature's column so it carries no information, and measure how much performance (AUC) drops. A large drop means the feature is important.

**Q12. Why use ROC-AUC and not just accuracy?**
The data is imbalanced (65% non-diabetic). A model that always says "no diabetes" would get 65% accuracy but be useless. AUC measures how well the model ranks positive cases above negative ones, independent of the threshold.

**Q13. If the interpretable model is as good, why do people use black boxes?**
On images, audio and text, deep learning is far more accurate, and interpretable models struggle there. On small tabular datasets like this one, interpretable models are often just as good. This is Rudin's argument.

**Q14. Why did the black box not win?**
The dataset is small (576 training rows) and the relationships are fairly simple and smooth. Complex models can overfit small data, while the GAM's smoothness acts as regularisation.

**Q15. What are the ethical concerns?**
- **Fairness:** the data covers one ethnic group of women only.
- **Accountability:** a doctor must stay responsible, and explanations let them check the model.
- **Safety:** a hidden wrong pattern could harm patients.
- **Transparency:** patients deserve reasons.

**Q16. How do you know the web demo is correct?**
Python saves reference predictions for three test patients. The demo's predictions were compared with them and match to 4 decimal places for all four models.

**Q17. What are the "complexity" measures you used?**
Number of parameters (logistic regression: 7), number of rules and depth (tree: 8 rules, depth 3), number of shape functions (GAM: 6), and number of nodes (black box: 2,184).

**Q18. Can the decision tree's 0% leaf be trusted?**
It is a leaf with 29 training patients (Glucose ≤ 132.5, Age > 30.5, BMI ≤ 26.9), none of whom had diabetes. With such a small group, the true risk is low but not exactly zero. This shows that leaf sample size matters when reading a tree.

---

## 15. Key terms glossary

| Term | Meaning |
|---|---|
| **Black box** | A model whose internal logic is too complex for humans to understand |
| **Glass box** | An interpretable model whose logic can be read directly |
| **Ante-hoc** | Interpretable by design |
| **Post-hoc** | An explanation produced after training, from outside the model |
| **Global explanation** | How the model behaves across all data |
| **Local explanation** | Why the model made one specific prediction |
| **Log-odds** | log(p / (1 − p)); the scale on which logistic regression and GAMs are additive |
| **Sigmoid** | 1 / (1 + e⁻ᶻ); turns log-odds into a probability between 0 and 1 |
| **Odds ratio** | e^weight; the multiplier on the odds for a one-unit increase |
| **Shape function** | The curve fᵢ(x) a GAM learns for one feature |
| **Spline** | A smooth curve built from pieces of polynomials joined at knots |
| **Surrogate model** | A simple model trained to imitate a complex one |
| **Fidelity** | How closely an explanation agrees with the model it explains |
| **ROC-AUC** | Probability that the model ranks a random positive case above a random negative one |
| **Permutation importance** | The drop in performance when a feature's values are shuffled |

---

## 16. References

1. Molnar, C. (2023). *Interpretable Machine Learning: A Guide for Making Black Box Models Explainable* (2nd ed.). — Chapters on linear models, logistic regression, decision trees, GAMs, permutation importance and global surrogates.
2. Rudin, C. (2019). Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead. *Nature Machine Intelligence*, 1, 206–215.
3. Hastie, T., & Tibshirani, R. (1990). *Generalized Additive Models*. Chapman & Hall.
4. Lou, Y., Caruana, R., Gehrke, J., & Hooker, G. (2013). Accurate intelligible models with pairwise interactions (GA²M). *KDD*.
5. Breiman, L. (2001). Random forests. *Machine Learning*, 45, 5–32. (Origin of permutation importance.)
6. Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. *Annals of Statistics*, 29(5).
7. Doshi-Velez, F., & Kim, B. (2017). Towards a rigorous science of interpretable machine learning. *arXiv:1702.08608*.
8. Smith, J. W., Everhart, J. E., Dickson, W. C., Knowler, W. C., & Johannes, R. S. (1988). Using the ADAP learning algorithm to forecast the onset of diabetes mellitus. *Proceedings of the Annual Symposium on Computer Application in Medical Care*. (Pima Indians Diabetes dataset.)
9. Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.

---

*XAI Mini Project · Modules 1 & 2 · For teaching purposes only; not a medical diagnostic tool.*
