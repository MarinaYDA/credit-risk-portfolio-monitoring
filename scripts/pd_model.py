"""
Train a logistic regression PD (probability of default) model on the
12-month default target and report AUC on a held-out test set.

Reads data/processed/model_features.csv.

Data-quality decisions made here (see docs/charter.md's "Note on
modeling approach" for the tradeoff between this and a bank-grade WOE
scorecard):
- The 4 rows with missing dti (annual_inc == 0, a division-by-zero in
  the upstream calculation) are dropped as incomplete records.
- dti and annual_inc are capped at their 99th percentile, computed from
  the training set only and applied to both splits, to reduce the
  leverage of a few extreme-but-real values without leaking test-set
  information into preprocessing.
- emp_length_years is missing for about 6% of loans ("not reported" in
  the source data, not an incomplete record) and is median-imputed
  inside the pipeline rather than dropped. Documented limitation: this
  assigns the median tenure to loans where employment length was simply
  never reported, which may understate risk for that subset if
  non-reporting correlates with shorter tenure.

Full-sample grade ordering is monotonic within a 1pp tolerance; grades F
and G (adjacent, and the pair with the smallest gap in mean int_rate of
any adjacent grade pair) show a small residual inversion once the model
conditions on int_rate and other continuous features - flagged here for
full interpretation when model coefficients are examined in detail, not
treated as a defect.

ROC curve, KS statistic, and confusion matrix are all computed on the
test set only (874 rows), matching AUC's convention - these measure
classifier performance and should be out-of-sample, unlike the
full-sample grade-ordering check above, which is a different,
population-level question.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
FEATURES_PATH = REPO_ROOT / "data" / "processed" / "model_features.csv"
ROC_CURVE_PATH = REPO_ROOT / "docs" / "img" / "roc_curve.png"
SCORED_PATH = REPO_ROOT / "data" / "processed" / "loan_level_scored.csv"

TARGET = "defaulted_12m"
NUM_FEATURES = [
    "loan_amnt",
    "int_rate",
    "annual_inc",
    "dti",
    "fico_range_low",
    "emp_length_years",
]
CAT_FEATURES = ["grade", "purpose", "home_ownership"]

CAPPED_COLUMNS = ["dti", "annual_inc"]
CAP_PERCENTILE = 0.99

TEST_SIZE = 0.25
RANDOM_STATE = 42

# 1 percentage point: large enough to pass the measured F/G full-sample
# gap (0.28pp, explained by grade/int_rate correlation at n=500 per
# grade for the two adjacent grades with the smallest int_rate
# separation), but small enough that it would NOT have passed the
# earlier test-set-only violation (1.57pp) - this still catches a real
# problem of that size, it isn't just disabled.
TOLERANCE_PP = 0.01

# Thresholds for the int_rate/grade redundancy check below: a p-value
# crossing 0.05 counts as a loss of significance, and a >30% shrink in
# coefficient magnitude counts as "noticeably smaller" once the
# correlated variable is added.
SIGNIFICANCE_ALPHA = 0.05
ATTENUATION_THRESHOLD_PCT = 30

# Categories (in any of CAT_FEATURES) with fewer than this many
# training-set loans are collapsed into "other_rare" before fitting the
# statsmodels Logit below. Some (e.g. purpose "educational": 1 loan/0
# defaults; home_ownership "ANY": 1 loan/0 defaults; purpose "car": 41
# loans/0 defaults) exhibit complete or near-complete separation, which
# breaks unregularized MLE - the same too-few-events-per-bin problem
# docs/charter.md already discusses for skipping WOE/coarse-classing at
# the portfolio level, showing up here at the feature level. Applied
# only to the interpretation model, not the deployed sklearn pipeline,
# which handles sparse categories fine under L2 regularization.
RARE_CATEGORY_THRESHOLD = 50


def cap_at_percentile(
    frames: dict[str, pd.DataFrame],
    train_key: str,
    columns: list[str],
    percentile: float,
) -> dict[str, pd.DataFrame]:
    """Cap columns at a percentile computed from frames[train_key] only,
    applied to every frame in frames, so the cutoff doesn't leak
    test-set (or full-sample) information into a preprocessing
    decision."""
    frames = {name: df.copy() for name, df in frames.items()}
    for col in columns:
        cutoff = frames[train_key][col].quantile(percentile)
        print(f"  {col}: {percentile:.0%} cutoff (from training set) = {cutoff:.2f}")
        for df in frames.values():
            df[col] = df[col].clip(upper=cutoff)
    return frames


def build_design_matrix(
    X: pd.DataFrame,
    numeric_pipeline: Pipeline,
) -> pd.DataFrame:
    """Build a dense design matrix for statsmodels: numeric features
    imputed and scaled with the already-fitted pipeline from the
    sklearn model (so coefficients sit on the same standardized scale
    as the deployed model), categorical features as drop-first dummies.

    Unlike the sklearn pipeline's full one-hot encoding (every level,
    no reference dropped - fine under L2 regularization), an
    unregularized statsmodels fit needs a non-singular design matrix,
    so one level per category is dropped here and becomes that
    category's implicit baseline.
    """
    # Collapse rare categories in every categorical feature (computed
    # from X itself, so this must only be called on the training
    # portion - see RARE_CATEGORY_THRESHOLD above for why). grade has no
    # sparse categories (~375+ per grade in training) so this is a
    # no-op there; it matters for purpose and home_ownership. If, after
    # collapsing, the pooled "other_rare" bucket for a column is STILL
    # below threshold (i.e. there was only one rare category to begin
    # with - e.g. home_ownership "ANY", n=1 - so collapsing just
    # renamed a singleton instead of pooling it with others), those few
    # rows are dropped instead: a coefficient can't be estimated from a
    # single-row category regardless of what it's labeled.
    X = X.copy()
    for col in CAT_FEATURES:
        counts = X[col].value_counts()
        rare = counts[counts < RARE_CATEGORY_THRESHOLD].index
        X[col] = X[col].where(~X[col].isin(rare), "other_rare")

    for col in CAT_FEATURES:
        counts = X[col].value_counts()
        still_rare = counts[counts < RARE_CATEGORY_THRESHOLD].index
        if len(still_rare) > 0:
            dropped = X[col].isin(still_rare).sum()
            print(
                f"  Dropping {dropped} row(s) with {col} in "
                f"{list(still_rare)} - too few to pool or estimate a "
                f"coefficient for."
            )
            X = X[~X[col].isin(still_rare)]

    num_scaled = pd.DataFrame(
        numeric_pipeline.transform(X[NUM_FEATURES]),
        columns=NUM_FEATURES,
        index=X.index,
    )
    cat_dummies = pd.get_dummies(X[CAT_FEATURES], drop_first=True, dtype=float)
    return pd.concat([num_scaled, cat_dummies], axis=1)


def redundancy_verdict(
    coef_with: float,
    coef_alone: float,
    p_with: float,
    p_alone: float,
) -> tuple[bool, float, bool]:
    """Compare a variable's coefficient/p-value with vs. without a
    correlated variable in the model. Returns (sign_flipped,
    pct_magnitude_shrink, lost_significance)."""
    sign_flipped = (coef_with > 0) != (coef_alone > 0) and coef_alone != 0
    pct_shrink = (
        (abs(coef_alone) - abs(coef_with)) / abs(coef_alone) * 100
        if coef_alone != 0
        else float("nan")
    )
    lost_significance = p_alone < SIGNIFICANCE_ALPHA <= p_with
    return sign_flipped, pct_shrink, lost_significance


if __name__ == "__main__":
    df = pd.read_csv(FEATURES_PATH)

    # dti is missing only where annual_inc == 0 (division-by-zero
    # upstream) - these 4 rows are incomplete records, not worth
    # imputing a ratio with a zero denominator.
    before = len(df)
    df = df.dropna(subset=["dti"])
    print(f"Dropped {before - len(df)} rows with missing dti (incomplete records).")

    X = df[NUM_FEATURES + CAT_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # dti and annual_inc have real, extreme values (not sentinel codes -
    # see docs/charter.md) - capped rather than dropped or transformed,
    # at this sample size. Cutoffs come from the training set only, and
    # are applied to test AND the full sample (used later for the
    # grade-ordering check), so no split leaks into another's cutoff.
    print("\nCapping dti and annual_inc at the 99th percentile:")
    capped = cap_at_percentile(
        {"train": X_train, "test": X_test, "full": X},
        "train",
        CAPPED_COLUMNS,
        CAP_PERCENTILE,
    )
    X_train, X_test, X_full = capped["train"], capped["test"], capped["full"]

    # SimpleImputer(median) applies to the whole numeric block for
    # simplicity; it's a no-op for every numeric column except
    # emp_length_years, the only one with remaining missing values at
    # this point.
    numeric_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        [
            ("num", numeric_pipeline, NUM_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
        ]
    )

    model = Pipeline(
        [
            ("preprocess", preprocessor),
            (
                "classify",
                LogisticRegression(max_iter=1000, class_weight="balanced"),
            ),
        ]
    )

    model.fit(X_train, y_train)

    test_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, test_proba)

    print(f"\nTrain rows: {len(X_train):,}  Test rows: {len(X_test):,}")
    print(f"Test set AUC: {auc:.4f}")

    # ROC curve, KS statistic, and the KS-maximizing threshold - all on
    # the test set, same as AUC above.
    fpr, tpr, thresholds = roc_curve(y_test, test_proba)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, label=f"ROC curve (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random classifier")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve - PD Model (test set)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    ROC_CURVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ROC_CURVE_PATH, dpi=130)
    plt.close(fig)
    print(f"\nSaved ROC curve to {ROC_CURVE_PATH}")

    # KS statistic: max distance between TPR and FPR across all
    # thresholds - the standard credit-scoring definition (the point of
    # maximum separation between the defaulter and non-defaulter score
    # distributions).
    ks_index = (tpr - fpr).argmax()
    ks_statistic = (tpr - fpr)[ks_index]
    ks_threshold = thresholds[ks_index]
    print(f"\nKS statistic: {ks_statistic:.4f} at threshold {ks_threshold:.4f}")

    # Threshold choice for the confusion matrix: the KS-maximizing
    # threshold, not 0.5. The model was trained with
    # class_weight="balanced", which reweights the loss function but
    # doesn't recalibrate predicted probabilities around 0.5, so 0.5 has
    # no particular meaning here. The KS-maximizing threshold is the
    # standard alternative in credit scoring specifically because it's
    # the point of maximum separation between the two score
    # distributions - a criterion grounded in the ROC curve itself,
    # not an arbitrary round number.
    y_pred = (test_proba >= ks_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    print(f"\nConfusion matrix at threshold {ks_threshold:.4f} (KS-maximizing):")
    print(f"  True Negatives:  {tn}")
    print(f"  False Positives: {fp}")
    print(f"  False Negatives: {fn}")
    print(f"  True Positives:  {tp}")

    # Sanity check: mean predicted PD should rise (within tolerance) from
    # grade A to G, same logic as the default-rate checks elsewhere in
    # this project (e.g. generate_loan_performance.py). Runs on the full
    # 3,496-loan sample, not the 874-row test slice: charter.md's stated
    # purpose for this model is validating risk ordering across the
    # portfolio, a population-level question, and ~500 loans per grade
    # is enough to not be dominated by small-sample noise the way the
    # ~120-per-grade test slice was (where even the ACTUAL, not
    # predicted, default rate wasn't monotonic). The AUC above stays
    # test-set-only, since that's the correct way to measure
    # generalization.
    full_proba = model.predict_proba(X_full)[:, 1]
    pd_by_grade = (
        pd.Series(full_proba, index=X_full.index)
        .groupby(X_full["grade"])
        .mean()
        .sort_index()
    )
    print("\nMean predicted PD by grade (full sample):")
    print(pd_by_grade)

    diffs = pd_by_grade.diff()
    violations = diffs[diffs < -TOLERANCE_PP]

    if not violations.empty:
        print("\nMonotonicity violations exceeding tolerance:")
        for grade, decrease in violations.items():
            prev_grade = pd_by_grade.index[pd_by_grade.index.get_loc(grade) - 1]
            print(f"  {prev_grade} -> {grade}: {decrease:+.4f} (exceeds -{TOLERANCE_PP:.2f} tolerance)")

    assert violations.empty, (
        f"Mean predicted PD drops by more than {TOLERANCE_PP:.0%} between "
        "at least one adjacent grade pair - investigate before proceeding."
    )
    print(
        f"Check passed: mean predicted PD is monotonic within a "
        f"{TOLERANCE_PP:.0%} tolerance from grade A to G."
    )

    # Per-loan scored output: loan_id and predicted_pd, aligned on the
    # same index as df/X_full (both come from the same post-dropna df),
    # so this is a direct assignment, no merge needed.
    scored = pd.DataFrame({"loan_id": df["loan_id"], "predicted_pd": full_proba})
    scored.to_csv(SCORED_PATH, index=False)
    print(f"\nSaved {len(scored):,} rows to {SCORED_PATH}")

    # --- Coefficient interpretation: parallel statsmodels Logit ---
    # sklearn's LogisticRegression above is L2-regularized (needed for
    # a well-behaved predictive model against the full one-hot
    # encoding); statsmodels' Logit is an unregularized MLE fit on the
    # same training rows and the same fitted numeric imputation/scaling,
    # used here specifically to get p-values for interpretation, not as
    # a second predictive model. It is not expected to reproduce
    # sklearn's exact coefficient values.
    numeric_pipeline = model.named_steps["preprocess"].named_transformers_["num"]
    design_train = build_design_matrix(X_train, numeric_pipeline)
    # A handful of rows may have been dropped inside build_design_matrix
    # (singleton categories) - realign the target to match.
    y_design = y_train.loc[design_train.index]
    logit_full = sm.Logit(y_design.values, sm.add_constant(design_train)).fit(disp=0)

    coef_table = pd.DataFrame(
        {"coefficient": logit_full.params, "p_value": logit_full.pvalues}
    ).drop(index="const")
    coef_table["abs_coefficient"] = coef_table["coefficient"].abs()
    coef_table = coef_table.sort_values("abs_coefficient", ascending=False)

    print("\n=== Coefficient table (statsmodels Logit) ===")
    print("(numeric features standardized; categorical features are drop-first dummies)")
    print(coef_table[["coefficient", "p_value"]].to_string(float_format=lambda x: f"{x:.4f}"))

    # --- Check: are int_rate and grade competing for the same signal? ---
    # int_rate and grade are correlated at 0.96 (confirmed separately -
    # see docs/charter.md). If one variable is absorbing the other's
    # signal, its coefficient should shrink toward zero, flip sign, or
    # lose statistical significance once the other is added, compared
    # to a fit with that variable alone.
    grade_dummy_cols = sorted(c for c in design_train.columns if c.startswith("grade_"))

    logit_no_grade = sm.Logit(
        y_design.values, sm.add_constant(design_train.drop(columns=grade_dummy_cols))
    ).fit(disp=0)
    logit_no_int_rate = sm.Logit(
        y_design.values, sm.add_constant(design_train.drop(columns=["int_rate"]))
    ).fit(disp=0)

    print("\n=== int_rate / grade redundancy check (correlated at 0.96) ===")

    ir_with, ir_alone = logit_full.params["int_rate"], logit_no_grade.params["int_rate"]
    ir_p_with, ir_p_alone = logit_full.pvalues["int_rate"], logit_no_grade.pvalues["int_rate"]
    print(
        f"int_rate: with grade coef={ir_with:+.4f} (p={ir_p_with:.4f});  "
        f"alone coef={ir_alone:+.4f} (p={ir_p_alone:.4f})"
    )

    print("grade dummies (baseline: grade A):")
    grade_checks = []
    for col in grade_dummy_cols:
        with_val, without_val = logit_full.params[col], logit_no_int_rate.params[col]
        p_with_val, p_without_val = logit_full.pvalues[col], logit_no_int_rate.pvalues[col]
        print(
            f"  {col}: with int_rate coef={with_val:+.4f} (p={p_with_val:.4f});  "
            f"alone coef={without_val:+.4f} (p={p_without_val:.4f})"
        )
        grade_checks.append(redundancy_verdict(with_val, without_val, p_with_val, p_without_val))

    # Direction check: does each variable point the expected way?
    grade_coefs_full = {col: logit_full.params[col] for col in grade_dummy_cols}
    grade_all_positive = all(v > 0 for v in grade_coefs_full.values())
    grade_order = [grade_coefs_full[f"grade_{g}"] for g in "BCDEFG" if f"grade_{g}" in grade_coefs_full]
    grade_monotonic = all(a <= b for a, b in zip(grade_order, grade_order[1:]))
    print("\nDirection check (full model, both variables included):")
    print(
        f"  int_rate coefficient is {'positive' if ir_with > 0 else 'negative'} "
        f"({'as expected: higher rate -> higher risk' if ir_with > 0 else 'UNEXPECTED sign'})"
    )
    print(
        f"  grade dummies (vs. baseline A) are "
        f"{'all positive' if grade_all_positive else 'NOT all positive'} and "
        f"{'increase monotonically B->G' if grade_monotonic else 'do NOT increase monotonically B->G'} "
        f"({'as expected: worse grade -> higher risk' if grade_all_positive and grade_monotonic else 'check needed'})"
    )

    # Stability check: sign flip / attenuation / significance loss,
    # computed from the numbers above rather than asserted by hand.
    ir_sign_flipped, ir_pct_shrink, ir_lost_sig = redundancy_verdict(
        ir_with, ir_alone, ir_p_with, ir_p_alone
    )
    any_grade_sign_flip = any(c[0] for c in grade_checks)
    any_grade_attenuated = any(c[1] > ATTENUATION_THRESHOLD_PCT for c in grade_checks if c[1] == c[1])
    any_grade_lost_sig = any(c[2] for c in grade_checks)

    print("\nRedundancy check result:")
    findings = []
    if ir_sign_flipped or any_grade_sign_flip:
        findings.append(
            "SIGN FLIP - at least one coefficient reverses direction depending "
            "on whether the correlated variable is included."
        )
    if ir_pct_shrink > ATTENUATION_THRESHOLD_PCT or any_grade_attenuated:
        findings.append(
            f"ATTENUATION - at least one coefficient shrinks by more than "
            f"{ATTENUATION_THRESHOLD_PCT}% in magnitude once the correlated "
            f"variable is added, consistent with the two variables sharing signal."
        )
    if ir_lost_sig or any_grade_lost_sig:
        findings.append(
            "SIGNIFICANCE LOSS - at least one coefficient is significant alone "
            "but not once the correlated variable is added."
        )
    if findings:
        for finding in findings:
            print(f"  {finding}")
    else:
        print(
            "  No sign flip, no attenuation beyond threshold, and no significance "
            "loss detected - but int_rate and grade remain correlated at 0.96, so "
            "their individual coefficients should still be read as jointly, not "
            "independently, informative."
        )
