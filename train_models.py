"""
Glass Box vs Black Box - XAI mini project (Modules 1 and 2)
-----------------------------------------------------------
Dataset : Pima Indians Diabetes (768 patients, 8 clinical measurements)
Task    : predict whether a patient has diabetes

Models
  * Black box  : Gradient Boosting (150 trees)               -> needs post-hoc XAI
  * Ante-hoc 1 : Logistic Regression (coefficients)          -> Module 2
  * Ante-hoc 2 : Decision Tree, depth 3 (IF-THEN rules)      -> Module 2
  * Ante-hoc 3 : Generalized Additive Model (spline GAM)     -> Module 2
Evaluation
  * Accuracy / ROC-AUC on a held-out test set
  * Complexity (number of parameters / rules)
  * Fidelity of a surrogate tree that imitates the black box  -> Module 2

Run:  python train_models.py      (writes models.json for the web demo)
"""
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import SplineTransformer, StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

RAW_COLS = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigree", "Age", "Outcome"]
FEATURES = ["Pregnancies", "Glucose", "BloodPressure", "BMI", "DiabetesPedigree", "Age"]
LABELS = {
    "Pregnancies": "Pregnancies",
    "Glucose": "Glucose (mg/dL)",
    "BloodPressure": "Blood pressure (mm Hg)",
    "BMI": "BMI (kg/m²)",
    "DiabetesPedigree": "Family history score",
    "Age": "Age (years)",
}

# ---------------------------------------------------------------- data
df = pd.read_csv("diabetes.csv", header=None, names=RAW_COLS)
X = df[FEATURES].astype(float).copy()
y = df["Outcome"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42)

# A value of 0 for glucose, blood pressure or BMI means "not recorded".
medians = {}
for col in ["Glucose", "BloodPressure", "BMI"]:
    medians[col] = float(X_train.loc[X_train[col] > 0, col].median())
    X_train.loc[X_train[col] == 0, col] = medians[col]
    X_test.loc[X_test[col] == 0, col] = medians[col]

Xtr, Xte = X_train.values, X_test.values


def scores(p_test):
    return {"accuracy": round(accuracy_score(y_test, p_test >= 0.5), 3),
            "auc": round(roc_auc_score(y_test, p_test), 3)}


# ---------------------------------------------------------------- black box
gbm = GradientBoostingClassifier(n_estimators=150, max_depth=3,
                                 learning_rate=0.05, random_state=0)
gbm.fit(Xtr, y_train)
p_gbm = gbm.predict_proba(Xte)[:, 1]


def tree_to_dict(t, value_fn):
    tr = t.tree_
    return {"f": tr.feature.tolist(), "t": [round(v, 4) for v in tr.threshold],
            "l": tr.children_left.tolist(), "r": tr.children_right.tolist(),
            "v": [round(value_fn(i), 5) for i in range(tr.node_count)]}


gbm_trees = [tree_to_dict(est[0], lambda i, e=est[0]: float(e.tree_.value[i, 0, 0]))
             for est in gbm.estimators_]
# raw score = init + learning_rate * sum(tree outputs)
x0 = Xtr[:1]
tree_sum = sum(est[0].predict(x0)[0] for est in gbm.estimators_)
gbm_init = float(gbm.decision_function(x0)[0] - gbm.learning_rate * tree_sum)

perm = permutation_importance(gbm, Xte, y_test, scoring="roc_auc",
                              n_repeats=20, random_state=0)

# ---------------------------------------------------------------- logistic regression
scaler = StandardScaler().fit(Xtr)
lr = LogisticRegression(C=1.0, max_iter=1000).fit(scaler.transform(Xtr), y_train)
p_lr = lr.predict_proba(scaler.transform(Xte))[:, 1]

# ---------------------------------------------------------------- decision tree
tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=20, random_state=0)
tree.fit(Xtr, y_train)
p_tree = tree.predict_proba(Xte)[:, 1]
tr = tree.tree_
tree_dict = tree_to_dict(tree, lambda i: float(tr.value[i, 0, 1] / tr.value[i, 0].sum()))
tree_dict["n"] = tr.n_node_samples.tolist()
print(export_text(tree, feature_names=FEATURES))

# ---------------------------------------------------------------- GAM
# g(P(diabetes)) = b0 + f1(x1) + f2(x2) + ... where each f is a cubic spline.
splines = [SplineTransformer(n_knots=5, degree=3).fit(Xtr[:, [j]]) for j in range(len(FEATURES))]


def gam_basis(Xm):
    return np.hstack([s.transform(Xm[:, [j]]) for j, s in enumerate(splines)])


gam = LogisticRegression(C=0.5, max_iter=5000).fit(gam_basis(Xtr), y_train)
p_gam = gam.predict_proba(gam_basis(Xte))[:, 1]

gam_shapes, offset, k = [], 0, 0
intercept = float(gam.intercept_[0])
for j, s in enumerate(splines):
    nb = s.transform(Xtr[:1, [j]]).shape[1]
    beta = gam.coef_[0, k:k + nb]
    k += nb
    lo, hi = Xtr[:, j].min(), Xtr[:, j].max()
    grid = np.linspace(lo, hi, 80)
    f_grid = s.transform(grid.reshape(-1, 1)) @ beta
    centre = float((s.transform(Xtr[:, [j]]) @ beta).mean())  # mean contribution = 0
    intercept += centre
    gam_shapes.append({"x": [round(v, 4) for v in grid],
                       "f": [round(v - centre, 4) for v in f_grid]})

# ---------------------------------------------------------------- fidelity of surrogate trees
gbm_train_labels = gbm.predict(Xtr)
gbm_test_labels = gbm.predict(Xte)
fidelity = []
for d in range(1, 9):
    sur = DecisionTreeClassifier(max_depth=d, random_state=0).fit(Xtr, gbm_train_labels)
    fidelity.append({"depth": d, "leaves": int(sur.get_n_leaves()),
                     "fidelity": round(accuracy_score(gbm_test_labels, sur.predict(Xte)), 3),
                     "accuracy": round(accuracy_score(y_test, sur.predict(Xte)), 3)})

# ---------------------------------------------------------------- example patients
def pick(mask):
    idx = np.where(mask)[0]
    return int(idx[0]) if len(idx) else 0

disagree = (p_gbm >= 0.5) != (p_tree >= 0.5)
examples = {
    "high": pick((p_gbm > 0.8) & (y_test == 1)),
    "low": pick((p_gbm < 0.1) & (y_test == 0)),
    "disagree": pick(disagree),
}
example_rows = {k: {"x": [round(float(v), 3) for v in Xte[i]], "label": int(y_test[i])}
                for k, i in examples.items()}

# ---------------------------------------------------------------- export
n_gbm_nodes = sum(len(t["f"]) for t in gbm_trees)
out = {
    "features": FEATURES,
    "labels": [LABELS[f] for f in FEATURES],
    "ranges": [[float(X[f][X[f] > 0].min() if f in medians else X[f].min()), float(X[f].max())] for f in FEATURES],
    "medians": medians,
    "n_train": int(len(y_train)), "n_test": int(len(y_test)),
    "prevalence": round(float(y.mean()), 3),
    "gbm": {"init": gbm_init, "lr": gbm.learning_rate, "trees": gbm_trees,
            "perm": [round(float(v), 4) for v in perm.importances_mean],
            "perm_sd": [round(float(v), 4) for v in perm.importances_std]},
    "logreg": {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
               "coef": lr.coef_[0].tolist(),
               "intercept": float(lr.intercept_[0])},
    "tree": tree_dict,
    "gam": {"intercept": intercept, "shapes": gam_shapes},
    "metrics": {
        "gbm": {**scores(p_gbm), "complexity": f"150 trees · {n_gbm_nodes} nodes"},
        "logreg": {**scores(p_lr), "complexity": "7 numbers (6 weights + 1 bias)"},
        "tree": {**scores(p_tree), "complexity": f"{tree.get_n_leaves()} rules, depth 3"},
        "gam": {**scores(p_gam), "complexity": "6 shape curves"},
    },
    "fidelity": fidelity,
    "examples": example_rows,
}
with open("models.json", "w") as fh:
    json.dump(out, fh, separators=(",", ":"))

for name, m in out["metrics"].items():
    print(f"{name:7s} acc={m['accuracy']:.3f} auc={m['auc']:.3f}  {m['complexity']}")
print("fidelity:", fidelity)
print("examples:", example_rows)
# reference predictions so the web demo can be checked against Python
ref = {k: {"gbm": float(gbm.predict_proba(Xte[[i]])[0, 1]),
           "logreg": float(lr.predict_proba(scaler.transform(Xte[[i]]))[0, 1]),
           "tree": float(tree.predict_proba(Xte[[i]])[0, 1]),
           "gam": float(gam.predict_proba(gam_basis(Xte[[i]]))[0, 1])} for k, i in examples.items()}
print("reference:", json.dumps(ref))
json.dump(ref, open("reference_predictions.json", "w"))
