import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
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


vectors_types = ["avg", "cov", "avg_cov", "tfidf", "max_pool"]
vectors = {}
for vector_type in vectors_types:
    filepath = os.path.join(cache.FOLDER_PATH, f"4_songs_vectors_{vector_type}.pkl")
    if os.path.exists(filepath):
        with open(filepath, "rb") as f:
            vectors[vector_type] = pickle.load(f)


def evaluate_classifier(X, y, classifier_name, classifier):
    X = processing.scale_vectors(X)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(classifier, X, y, cv=skf, scoring="accuracy")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    detail_results = {
        "cv_accuracy_mean": cv_scores.mean(),
        "cv_accuracy_std": cv_scores.std(),
        "test_accuracy": accuracy_score(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
    }

    return classifier, cv_scores.mean(), detail_results


results = {}
detailed_results = {}


composers = dataset["canonical_composer"].to_list()
unique_composers = sorted(set(composers))
print(f"Number of unique composers: {len(unique_composers)}")
print(f"Classes: {unique_composers}")


classifiers = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(kernel="rbf", C=10, gamma="scale", random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0, random_state=42),
}

trained_classifiers = {}


for vector_type, X in vectors.items():
    results[vector_type] = {}
    detailed_results[vector_type] = {}

    for cls_name, classifier in classifiers.items():
        trained_classifier, accuracy, details = evaluate_classifier(
            X, composers, cls_name, classifier
        )
        results[vector_type][cls_name] = accuracy
        detailed_results[vector_type][cls_name] = details
        trained_classifiers[cls_name] = trained_classifier


summary = pd.DataFrame(results)
print("\nSummary of classification accuracies:")
print(summary)

summary_csv_path = os.path.join(cache.FOLDER_PATH, "classification_summary.csv")
summary.to_csv(summary_csv_path)
print(f"Summary table saved to: {summary_csv_path}")


best_accuracy = 0
best_vector = None
best_classifier = None

for vector_type, classifiers in results.items():
    for cls_name, accuracy in classifiers.items():
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_vector = vector_type
            best_classifier = cls_name


print(f"\nBest accuracy: {best_accuracy:.4f}")
print(f"Best vector type: {best_vector}")
print(f"Best classifier: {best_classifier}")


best_model = trained_classifiers[best_classifier]
best_X = vectors[best_vector]
best_X_scaled = processing.scale_vectors(best_X)


X_train, X_test, y_train, y_test = train_test_split(
    best_X_scaled, composers, test_size=0.3, random_state=42, stratify=composers
)

best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)


cm = confusion_matrix(y_test, y_pred)
cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(12, 10))
plt.imshow(cm_normalized, interpolation="nearest", cmap=plt.cm.Blues)
plt.title(f"Normalized Confusion Matrix: {best_vector} + {best_classifier}")
plt.colorbar()
tick_marks = np.arange(len(unique_composers))
plt.xticks(tick_marks, unique_composers, rotation=90)
plt.yticks(tick_marks, unique_composers)
thresh = cm_normalized.max() / 2.0
for i, j in np.ndindex(cm_normalized.shape):
    plt.text(
        j,
        i,
        f"{cm_normalized[i, j]:.2f}",
        horizontalalignment="center",
        color="white" if cm_normalized[i, j] > thresh else "black",
    )
plt.tight_layout()
plt.ylabel("True Composer")
plt.xlabel("Predicted Composer")
cm_plot_path = os.path.join(
    cache.FOLDER_PATH, f"confusion_matrix_{best_vector}_{best_classifier.replace(' ', '_')}.png"
)
plt.savefig(cm_plot_path)
plt.close()
print(f"Confusion matrix saved to: {cm_plot_path}")
