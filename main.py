import os
import pandas as pd
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline


PATHNAME = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# load dataset (train and test dataset)
train_df = pd.read_csv("dataset/train.csv", index_col="id")
X_train = train_df.iloc[:, 0:-1]
y_train = train_df["target"]

test_df = pd.read_csv("dataset/test.csv", index_col="id")
X_test = test_df.iloc[:, 0:]

# pipelines
model_pipeline = Pipeline(steps=[("algo", LogisticRegression(n_jobs=-1))], verbose=1)

params = {
    "algo__penalty": ["l2", "elasticnet", "none"],
    "algo__dual": [True, False],
    "algo__fit_intercept": [True, False],
    "algo__C": [0.125, 0.25, 0.5, 1.0, 2.0],
    "algo__solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
    "algo__l1_ratio": [0.125, 0.25, 0.5, 1.0, 2.0, None],
}

model = GridSearchCV(
    estimator=model_pipeline,
    param_grid=params,
    cv=5,
    scoring=("roc_auc"),
    n_jobs=-1,
    verbose=1,
    return_train_score=True,
)

model.fit(X_train, y_train)

# save prediction
os.makedirs(f"submission/{PATHNAME}", exist_ok=True)
preds = model.predict(X_test)
preds_df = pd.DataFrame({"id": X_test.index, "target": preds})
preds_df.to_csv(f"submission/{PATHNAME}/preds.csv", index=False)

# save cv report
pd.DataFrame(model.cv_results_).sort_values(by="rank_test_score").to_csv(
    f"submission/{PATHNAME}/cv_result.csv", index=False
)
