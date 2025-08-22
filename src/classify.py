import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

import data
import cache
import processing


dataset = data.filter_dataset(data.new_dataset())


vectors_types = ["avg", "std", "avg_std", "tfidf", "max_pool"]
vectors = {}
for vector_type in vectors_types:
    filepath = os.path.join(cache.FOLDER_PATH, f"4_songs_vectors_{vector_type}.pkl")
    if os.path.exists(filepath):
        with open(filepath, "rb") as f:
            vectors[vector_type] = pickle.load(f)


def evaluate_classifier_cv(X, y, classifier_name, classifier):
    """Evaluate classifier using only cross-validation approaches"""
    X = processing.scale_vectors(X)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Define scoring metrics for cross-validation
    scoring = ["accuracy", "precision_weighted", "recall_weighted", "f1_weighted"]

    # Perform cross-validation with multiple metrics
    cv_results = cross_validate(classifier, X, y, cv=skf, scoring=scoring)

    # Compile results
    results = {
        "accuracy_mean": cv_results["test_accuracy"].mean(),
        "precision_mean": cv_results["test_precision_weighted"].mean(),
        "recall_mean": cv_results["test_recall_weighted"].mean(),
        "f1_mean": cv_results["test_f1_weighted"].mean(),
    }

    return results


# Initialize result storage
cv_results = {}

# Get composer labels
composers = dataset["canonical_composer"].to_list()
unique_composers = sorted(set(composers))
print(f"Number of unique composers: {len(unique_composers)}")
print(f"Classes: {unique_composers}")

# Define classifiers
classifiers = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(kernel="rbf", C=10, gamma="scale", random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0, random_state=42),
}

# Evaluate all combinations using cross-validation
print("\n" + "=" * 80)
print("CROSS-VALIDATION EVALUATION")
print("=" * 80)

for vector_type, X in vectors.items():
    cv_results[vector_type] = {}
    print(f"\n{'=' * 20} VECTOR TYPE: {vector_type.upper()} {'=' * 20}")

    for cls_name, classifier in classifiers.items():
        print(f"\nEvaluating {cls_name}...")
        results = evaluate_classifier_cv(X, composers, cls_name, classifier)
        cv_results[vector_type][cls_name] = results

        # Display results immediately
        print(f"--- {cls_name} Results ---")
        print(f"Accuracy:  {results['accuracy_mean']:.4f}")
        print(f"Precision: {results['precision_mean']:.4f}")
        print(f"Recall:    {results['recall_mean']:.4f}")
        print(f"F1-Score:  {results['f1_mean']:.4f}")

# Create comprehensive summary tables
print(f"\n{'=' * 80}")
print("COMPREHENSIVE CROSS-VALIDATION RESULTS")
print("=" * 80)

# Prepare data for summary tables
all_results = []
for vector_type in vectors_types:
    if vector_type in cv_results:
        for cls_name in classifiers.keys():
            if cls_name in cv_results[vector_type]:
                results = cv_results[vector_type][cls_name]
                all_results.append(
                    {
                        "Vector_Type": vector_type,
                        "Classifier": cls_name,
                        "Accuracy_Mean": results["accuracy_mean"],
                        "Precision_Mean": results["precision_mean"],
                        "Recall_Mean": results["recall_mean"],
                        "F1_Mean": results["f1_mean"],
                    }
                )

# Create and display comprehensive DataFrame
if all_results:
    results_df = pd.DataFrame(all_results)

    # Display full results table
    print("\nFull Results Table:")
    print(results_df.to_string(index=False, float_format="%.4f"))

    # Save comprehensive results
    comprehensive_csv_path = os.path.join(cache.FOLDER_PATH, "cv_comprehensive_results.csv")
    results_df.to_csv(comprehensive_csv_path, index=False)
    print(f"\nComprehensive CV results saved to: {comprehensive_csv_path}")

    # Create separate summary tables for each metric
    print(f"\n{'=' * 60}")
    print("ACCURACY SUMMARY")
    print("=" * 60)
    accuracy_pivot = results_df.pivot(
        index="Vector_Type", columns="Classifier", values="Accuracy_Mean"
    )
    print(accuracy_pivot.to_string(float_format="%.4f"))

    print(f"\n{'=' * 60}")
    print("PRECISION SUMMARY")
    print("=" * 60)
    precision_pivot = results_df.pivot(
        index="Vector_Type", columns="Classifier", values="Precision_Mean"
    )
    print(precision_pivot.to_string(float_format="%.4f"))

    print(f"\n{'=' * 60}")
    print("RECALL SUMMARY")
    print("=" * 60)
    recall_pivot = results_df.pivot(index="Vector_Type", columns="Classifier", values="Recall_Mean")
    print(recall_pivot.to_string(float_format="%.4f"))

    print(f"\n{'=' * 60}")
    print("F1-SCORE SUMMARY")
    print("=" * 60)
    f1_pivot = results_df.pivot(index="Vector_Type", columns="Classifier", values="F1_Mean")
    print(f1_pivot.to_string(float_format="%.4f"))

    # Save individual metric summaries
    accuracy_pivot.to_csv(os.path.join(cache.FOLDER_PATH, "cv_accuracy_summary.csv"))
    precision_pivot.to_csv(os.path.join(cache.FOLDER_PATH, "cv_precision_summary.csv"))
    recall_pivot.to_csv(os.path.join(cache.FOLDER_PATH, "cv_recall_summary.csv"))
    f1_pivot.to_csv(os.path.join(cache.FOLDER_PATH, "cv_f1_summary.csv"))

    # Find best models for each metric
    print(f"\n{'=' * 80}")
    print("BEST MODELS BY METRIC (CROSS-VALIDATION)")
    print("=" * 80)

    best_accuracy = results_df.loc[results_df["Accuracy_Mean"].idxmax()]
    best_precision = results_df.loc[results_df["Precision_Mean"].idxmax()]
    best_recall = results_df.loc[results_df["Recall_Mean"].idxmax()]
    best_f1 = results_df.loc[results_df["F1_Mean"].idxmax()]

    print(
        f"Best Accuracy:  {best_accuracy['Accuracy_Mean']:.4f} ({best_accuracy['Vector_Type']} + {best_accuracy['Classifier']})"
    )
    print(
        f"Best Precision: {best_precision['Precision_Mean']:.4f} ({best_precision['Vector_Type']} + {best_precision['Classifier']})"
    )
    print(
        f"Best Recall:    {best_recall['Recall_Mean']:.4f} ({best_recall['Vector_Type']} + {best_recall['Classifier']})"
    )
    print(
        f"Best F1-Score:  {best_f1['F1_Mean']:.4f} ({best_f1['Vector_Type']} + {best_f1['Classifier']})"
    )

    # Use best F1-score model for final evaluation (most balanced metric)
    best_vector_type = best_f1["Vector_Type"]
    best_classifier_name = best_f1["Classifier"]

    print(f"\n{'=' * 80}")
    print("FINAL MODEL EVALUATION")
    print("=" * 80)
    print(f"Selected model: {best_vector_type} + {best_classifier_name} (Best F1-Score)")
    print(f"CV F1-Score: {best_f1['F1_Mean']:.4f}")

    # Train final model on full dataset for confusion matrix
    best_X = vectors[best_vector_type]
    best_X_scaled = processing.scale_vectors(best_X)
    best_classifier = classifiers[best_classifier_name]

    # Use train-test split only for confusion matrix visualization
    X_train, X_test, y_train, y_test = train_test_split(
        best_X_scaled, composers, test_size=0.3, random_state=42, stratify=composers
    )

    best_classifier.fit(X_train, y_train)
    y_pred = best_classifier.predict(X_test)

    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Convert to percentages
    cm_percent = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100

    plt.figure(figsize=(15, 15), facecolor="none")
    plt.imshow(cm_percent, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar()
    # Extract last names from composer names
    composer_last_names = [name.split()[-1] for name in unique_composers]

    tick_marks = np.arange(len(unique_composers))
    plt.xticks(tick_marks, composer_last_names, rotation=90, fontsize=18)
    plt.yticks(tick_marks, composer_last_names, fontsize=18)
    thresh = cm_percent.max() / 2.0
    for i, j in np.ndindex(cm_percent.shape):
        plt.text(
            j,
            i,
            f"{cm_percent[i, j]:.1f}%",
            horizontalalignment="center",
            color="white" if cm_percent[i, j] > thresh else "black",
            fontsize=18,
        )
    plt.tight_layout()
    plt.ylabel("True Composer", fontsize=24)
    plt.xlabel("Predicted Composer", fontsize=24)

    plt.gca().patch.set_alpha(0)

    cm_plot_path = os.path.join(
        cache.FOLDER_PATH,
        f"confusion_matrix_{best_vector_type}_{best_classifier_name.replace(' ', '_')}_cv.png",
    )
    plt.savefig(cm_plot_path)
    plt.close()
    print(f"Confusion matrix saved to: {cm_plot_path}")

    # Final test set performance (for reference only)
    test_accuracy = accuracy_score(y_test, y_pred)
    test_report = classification_report(y_test, y_pred, output_dict=True)

    print(f"\nTest set performance (single split, for reference):")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    if "weighted avg" in test_report:
        wa = test_report["weighted avg"]
        print(f"Test Precision: {wa['precision']:.4f}")
        print(f"Test Recall: {wa['recall']:.4f}")
        print(f"Test F1-Score: {wa['f1-score']:.4f}")

    print(f"\nNote: Cross-validation results above are more reliable for model comparison.")
    print(f"Test set results are shown for reference and confusion matrix generation only.")

else:
    print("No results to display. Check if vector files exist and are loaded correctly.")
