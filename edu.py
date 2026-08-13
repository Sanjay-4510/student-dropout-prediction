import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
    StratifiedKFold
)
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    ConfusionMatrixDisplay,
    roc_auc_score
)

from xgboost import XGBClassifier


warnings.filterwarnings("ignore")

DATA_FILE = "data.csv.xlsx"
RANDOM_STATE = 42


def load_data():

    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(
            f"'{DATA_FILE}' was not found. "
            "Place it in the same folder as edu.py."
        )

    df = pd.read_excel(DATA_FILE)

    if "Target" not in df.columns:
        raise ValueError(
            "The dataset must contain a 'Target' column."
        )

    return df


def print_data_summary(df):

    print("\nDataset loaded successfully!")
    print("Dataset shape:", df.shape)

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nMissing values:")
    print(df.isnull().sum().sum())

    print("\nDuplicate rows:")
    print(df.duplicated().sum())

    if df.isnull().sum().sum() > 0:
        raise ValueError(
            "Missing values were found in the dataset."
        )

    if df.duplicated().sum() > 0:
        print(
            "\nWarning: duplicate rows exist."
        )

    print("\nTarget distribution:")
    print(
        df["Target"].value_counts()
    )

    print("\nTarget percentage:")
    print(
        df["Target"]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
    )


def prepare_data(df):

    X = df.drop(
        "Target",
        axis=1
    )

    y_text = df["Target"]

    encoder = LabelEncoder()

    y = encoder.fit_transform(
        y_text
    )

    print("\nX shape:", X.shape)
    print("y shape:", y.shape)

    print("\nTarget mapping:")

    for number, name in enumerate(
        encoder.classes_
    ):
        print(
            f"{number} = {name}"
        )

    return X, y, encoder


def split_data(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print("\nTrain/Test split:")
    print(
        "X_train:",
        X_train.shape
    )

    print(
        "X_test:",
        X_test.shape
    )

    print(
        "y_train:",
        y_train.shape
    )

    print(
        "y_test:",
        y_test.shape
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test
    )


def build_baseline():

    return XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        random_state=RANDOM_STATE,
        eval_metric="mlogloss",
        tree_method="hist",
        n_jobs=-1
    )


def build_search_model():

    return XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        random_state=RANDOM_STATE,
        eval_metric="mlogloss",
        tree_method="hist",
        n_jobs=-1
    )


def evaluate_model(
    model,
    X_test,
    y_test,
    encoder,
    model_name
):

    predictions = model.predict(
        X_test
    )

    probabilities = model.predict_proba(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    balanced_accuracy = balanced_accuracy_score(
        y_test,
        predictions
    )

    macro_precision = precision_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    macro_recall = recall_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro"
    )

    weighted_f1 = f1_score(
        y_test,
        predictions,
        average="weighted"
    )

    try:

        roc_auc = roc_auc_score(
            y_test,
            probabilities,
            multi_class="ovr",
            average="macro"
        )

    except Exception:

        roc_auc = np.nan

    print("\n" + model_name)

    print(
        "\nAccuracy:",
        f"{accuracy * 100:.2f}%"
    )

    print(
        "Balanced Accuracy:",
        f"{balanced_accuracy * 100:.2f}%"
    )

    print(
        "Macro Precision:",
        f"{macro_precision:.4f}"
    )

    print(
        "Macro Recall:",
        f"{macro_recall:.4f}"
    )

    print(
        "Macro F1:",
        f"{macro_f1:.4f}"
    )

    print(
        "Weighted F1:",
        f"{weighted_f1:.4f}"
    )

    if not np.isnan(roc_auc):

        print(
            "Macro ROC-AUC:",
            f"{roc_auc:.4f}"
        )

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            target_names=encoder.classes_,
            digits=4
        )
    )

    matrix = confusion_matrix(
        y_test,
        predictions
    )

    print("Confusion Matrix:")
    print(matrix)

    metrics = {
        "Model": model_name,
        "Accuracy": accuracy,
        "Balanced Accuracy": balanced_accuracy,
        "Macro Precision": macro_precision,
        "Macro Recall": macro_recall,
        "Macro F1": macro_f1,
        "Weighted F1": weighted_f1,
        "Macro ROC-AUC": roc_auc
    }

    return (
        metrics,
        predictions,
        probabilities,
        matrix
    )


def plot_confusion_matrix(
    matrix,
    encoder,
    title
):

    fig, ax = plt.subplots(
        figsize=(7, 6)
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=encoder.classes_
    )

    display.plot(
        ax=ax,
        cmap="Blues",
        values_format="d"
    )

    ax.set_title(
        title
    )

    plt.tight_layout()
    plt.show()


def hyperparameter_search(
    X_train,
    y_train
):

    print(
        "\nStarting advanced XGBoost tuning..."
    )

    balanced_weights = compute_sample_weight(
        class_weight="balanced",
        y=y_train
    )

    model = build_search_model()

    parameter_grid = {

        "n_estimators": [
            200,
            300,
            400,
            500,
            700
        ],

        "max_depth": [
            3,
            4,
            5,
            6,
            7
        ],

        "learning_rate": [
            0.02,
            0.03,
            0.05,
            0.07,
            0.10
        ],

        "min_child_weight": [
            1,
            3,
            5,
            7
        ],

        "subsample": [
            0.70,
            0.80,
            0.90,
            1.00
        ],

        "colsample_bytree": [
            0.70,
            0.80,
            0.90,
            1.00
        ],

        "gamma": [
            0,
            0.05,
            0.10,
            0.30,
            0.50
        ],

        "reg_alpha": [
            0,
            0.01,
            0.10,
            0.50
        ],

        "reg_lambda": [
            1,
            2,
            5,
            10
        ]
    }

    cv = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=parameter_grid,
        n_iter=15,
        scoring="f1_macro",
        cv=cv,
        random_state=RANDOM_STATE,
        verbose=1,
        n_jobs=1,
        refit=True
    )

    search.fit(
        X_train,
        y_train,
        sample_weight=balanced_weights
    )

    print(
        "\nHyperparameter tuning completed!"
    )

    print(
        "\nBest cross-validation Macro F1:",
        f"{search.best_score_:.4f}"
    )

    print(
        "\nBest parameters:"
    )

    for parameter, value in (
        search.best_params_.items()
    ):

        print(
            f"{parameter}: {value}"
        )

    return search.best_estimator_, search


def print_top_features(
    model,
    X,
    number=20
):

    importance = model.feature_importances_

    importance_df = pd.DataFrame(
        {
            "Feature": X.columns,
            "Importance": importance
        }
    ).sort_values(
        "Importance",
        ascending=False
    )

    print(
        f"\nTop {number} XGBoost Features:"
    )

    print(
        importance_df.head(
            number
        ).to_string(index=False)
    )

    top = importance_df.head(
        15
    ).iloc[::-1]

    fig, ax = plt.subplots(
        figsize=(10, 7)
    )

    ax.barh(
        top["Feature"],
        top["Importance"]
    )

    ax.set_xlabel(
        "XGBoost Importance"
    )

    ax.set_title(
        "Top 15 XGBoost Features"
    )

    plt.tight_layout()
    plt.show()

    return importance_df


def convert_shap_output(
    shap_values,
    class_index,
    number_of_classes
):

    if isinstance(
        shap_values,
        list
    ):

        array = np.asarray(
            shap_values[class_index]
        )

        if array.ndim == 2:
            return array

        return array.reshape(
            array.shape[0],
            -1
        )

    array = np.asarray(
        shap_values
    )

    if array.ndim == 3:

        if array.shape[2] == number_of_classes:

            return array[
                :,
                :,
                class_index
            ]

        if array.shape[0] == number_of_classes:

            return array[
                class_index,
                :,
                :
            ]

    if array.ndim == 2:

        return array

    raise ValueError(
        f"Unsupported SHAP shape: {array.shape}"
    )


def run_shap_analysis(
    model,
    X_test,
    encoder
):

    print(
        "\nCreating SHAP explainer..."
    )

    explainer = shap.TreeExplainer(
        model
    )

    print(
        "Calculating SHAP values..."
    )

    shap_values = explainer.shap_values(
        X_test
    )

    print(
        "SHAP values calculated successfully!"
    )

    print(
        "SHAP output type:",
        type(shap_values)
    )

    if isinstance(
        shap_values,
        list
    ):

        print(
            "Number of class arrays:",
            len(shap_values)
        )

        for index, values in enumerate(
            shap_values
        ):

            print(
                f"Class {index} shape:",
                np.asarray(values).shape
            )

    else:

        print(
            "SHAP shape:",
            np.asarray(
                shap_values
            ).shape
        )

    dropout_index = list(
        encoder.classes_
    ).index(
        "Dropout"
    )

    dropout_values = convert_shap_output(
        shap_values,
        dropout_index,
        len(encoder.classes_)
    )

    print(
        "\nDropout SHAP matrix shape:",
        dropout_values.shape
    )

    print(
        "\nGenerating Dropout SHAP summary..."
    )

    shap.summary_plot(
        dropout_values,
        X_test,
        feature_names=X_test.columns,
        max_display=15
    )

    mean_abs = np.abs(
        dropout_values
    ).mean(
        axis=0
    )

    shap_importance = pd.DataFrame(
        {
            "Feature": X_test.columns,
            "Mean Absolute SHAP": mean_abs
        }
    ).sort_values(
        "Mean Absolute SHAP",
        ascending=False
    )

    print(
        "\nTop Dropout SHAP Features:"
    )

    print(
        shap_importance.head(
            15
        ).to_string(
            index=False
        )
    )

    return (
        explainer,
        shap_values,
        shap_importance
    )


def explain_student(
    model,
    explainer,
    student,
    encoder
):

    prediction = int(
        model.predict(
            student
        )[0]
    )

    probabilities = model.predict_proba(
        student
    )[0]

    predicted_label = encoder.inverse_transform(
        [prediction]
    )[0]

    individual_shap = explainer.shap_values(
        student
    )

    class_values = convert_shap_output(
        individual_shap,
        prediction,
        len(encoder.classes_)
    )

    individual_values = class_values[0]

    explanation = pd.DataFrame(
        {
            "Feature": student.columns,
            "Student Value": student.iloc[
                0
            ].values,
            "SHAP Value": individual_values
        }
    )

    explanation[
        "Absolute SHAP"
    ] = explanation[
        "SHAP Value"
    ].abs()

    explanation = explanation.sort_values(
        "Absolute SHAP",
        ascending=False
    )

    print(
        "\nIndividual Student Prediction:"
    )

    print(
        "Prediction:",
        predicted_label
    )

    for index, class_name in enumerate(
        encoder.classes_
    ):

        print(
            f"{class_name} probability:",
            f"{probabilities[index] * 100:.2f}%"
        )

    print(
        "\nTop individual factors:"
    )

    print(
        explanation.head(
            10
        ).to_string(
            index=False
        )
    )

    return (
        predicted_label,
        probabilities,
        explanation
    )


def save_artifacts(
    model,
    encoder,
    X,
    search
):

    joblib.dump(
        model,
        "dropout_xgboost_model.pkl"
    )

    joblib.dump(
        encoder,
        "target_encoder.pkl"
    )

    joblib.dump(
        list(X.columns),
        "feature_names.pkl"
    )

    joblib.dump(
        search.best_params_,
        "best_xgboost_params.pkl"
    )

    print(
        "\nSaved model artifacts:"
    )

    print(
        "dropout_xgboost_model.pkl"
    )

    print(
        "target_encoder.pkl"
    )

    print(
        "feature_names.pkl"
    )

    print(
        "best_xgboost_params.pkl"
    )


def main():

    print(
        "\nEDUPREDICT AI"
    )

    print(
        "Advanced Explainable Student Outcome Prediction"
    )

    df = load_data()

    print_data_summary(
        df
    )

    X, y, encoder = prepare_data(
        df
    )

    X_train, X_test, y_train, y_test = split_data(
        X,
        y
    )

    print(
        "\nTraining baseline model..."
    )

    baseline = build_baseline()

    baseline_weights = compute_sample_weight(
        class_weight="balanced",
        y=y_train
    )

    baseline.fit(
        X_train,
        y_train,
        sample_weight=baseline_weights
    )

    (
        baseline_metrics,
        baseline_pred,
        baseline_prob,
        baseline_cm
    ) = evaluate_model(
        baseline,
        X_test,
        y_test,
        encoder,
        "BASELINE XGBOOST"
    )

    print(
        "\nTraining tuned model..."
    )

    tuned_model, search = hyperparameter_search(
        X_train,
        y_train
    )

    tuned_weights = compute_sample_weight(
        class_weight="balanced",
        y=y_train
    )

    tuned_model.fit(
        X_train,
        y_train,
        sample_weight=tuned_weights
    )

    (
        tuned_metrics,
        tuned_pred,
        tuned_prob,
        tuned_cm
    ) = evaluate_model(
        tuned_model,
        X_test,
        y_test,
        encoder,
        "TUNED XGBOOST"
    )

    print(
        "\nModel comparison:"
    )

    comparison = pd.DataFrame(
        [
            baseline_metrics,
            tuned_metrics
        ]
    )

    print(
        comparison.to_string(
            index=False
        )
    )

    comparison.to_csv(
        "model_comparison.csv",
        index=False
    )

    print(
        "\nSaved: model_comparison.csv"
    )

    plot_confusion_matrix(
        tuned_cm,
        encoder,
        "Tuned XGBoost Confusion Matrix"
    )

    importance_df = print_top_features(
        tuned_model,
        X
    )

    (
        explainer,
        shap_values,
        shap_importance
    ) = run_shap_analysis(
        tuned_model,
        X_test,
        encoder
    )

    student_index = 0

    student = X_test.iloc[
        [student_index]
    ]

    predicted_label, probabilities, explanation = explain_student(
        tuned_model,
        explainer,
        student,
        encoder
    )

    base_dropout_probability = (
        baseline_prob[0]
        if baseline_prob.ndim > 1
        else 0
    )

    save_artifacts(
        tuned_model,
        encoder,
        X,
        search
    )

    final_results = {
        "Baseline Accuracy": baseline_metrics["Accuracy"],
        "Tuned Accuracy": tuned_metrics["Accuracy"],
        "Baseline Macro F1": baseline_metrics["Macro F1"],
        "Tuned Macro F1": tuned_metrics["Macro F1"],
        "Baseline Balanced Accuracy": baseline_metrics[
            "Balanced Accuracy"
        ],
        "Tuned Balanced Accuracy": tuned_metrics[
            "Balanced Accuracy"
        ],
        "Tuned Weighted F1": tuned_metrics[
            "Weighted F1"
        ],
        "Tuned Macro Precision": tuned_metrics[
            "Macro Precision"
        ],
        "Tuned Macro Recall": tuned_metrics[
            "Macro Recall"
        ],
        "Tuned Macro ROC-AUC": tuned_metrics[
            "Macro ROC-AUC"
        ]
    }

    results_df = pd.DataFrame(
        {
            "Metric": list(
                final_results.keys()
            ),
            "Value": list(
                final_results.values()
            )
        }
    )

    results_df.to_csv(
        "advanced_model_results.csv",
        index=False
    )

    print(
        "\nSaved: advanced_model_results.csv"
    )

    print(
        "\nFinal Summary"
    )

    print(
        "--------------------------------"
    )

    print(
        f"Baseline accuracy: "
        f"{baseline_metrics['Accuracy'] * 100:.2f}%"
    )

    print(
        f"Tuned accuracy: "
        f"{tuned_metrics['Accuracy'] * 100:.2f}%"
    )

    print(
        f"Baseline Macro F1: "
        f"{baseline_metrics['Macro F1']:.4f}"
    )

    print(
        f"Tuned Macro F1: "
        f"{tuned_metrics['Macro F1']:.4f}"
    )

    print(
        f"Final predicted student outcome: "
        f"{predicted_label}"
    )

    print(
        "\nTraining completed successfully."
    )


if __name__ == "__main__":

    main()
