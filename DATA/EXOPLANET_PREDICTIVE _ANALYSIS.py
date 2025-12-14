import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report, roc_curve, auc

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans


# Create plots directory
os.makedirs("plots", exist_ok=True)


# Load dataset
df = pd.read_csv(
    '/Users/uppupavankumar/Downloads/PREDICTIVE DATASET.csv',
    comment='#'
)

print(df.columns.tolist())


# Data cleaning
df = df.dropna()


# Target creation
df['planet_type'] = np.where(df['pl_rade'] < 2, 0, 1)


# Feature selection
X = df.drop('planet_type', axis=1)
y = df['planet_type']

X = X.select_dtypes(include=[np.number])
X = X.dropna()
y = y.loc[X.index]


# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)


# ---------------- FEATURE DISTRIBUTIONS ----------------

plt.figure()
plt.hist(df['pl_rade'], bins=30)
plt.xlabel("Planet Radius (Earth radii)")
plt.ylabel("Frequency")
plt.title("Distribution of Planet Radius")
plt.tight_layout()
plt.savefig("plots/feature_distribution_pl_rade.png", dpi=300, bbox_inches="tight")
plt.show()

plt.figure()
plt.hist(df['pl_bmasse'], bins=30)
plt.xlabel("Planet Mass (Earth mass)")
plt.ylabel("Frequency")
plt.title("Distribution of Planet Mass")
plt.tight_layout()
plt.savefig("plots/feature_distribution_pl_bmasse.png", dpi=300, bbox_inches="tight")
plt.show()


# ---------------- CORRELATION HEATMAP ----------------

corr_features = [
    'pl_rade',
    'pl_bmasse',
    'pl_orbper',
    'pl_orbsmax',
    'pl_eqt',
    'st_teff',
    'st_rad',
    'st_mass'
]

corr_data = df[corr_features].dropna()
corr_matrix = corr_data.corr()

plt.figure(figsize=(8, 6))
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    square=True
)
plt.title("Correlation Heatmap of Key Exoplanet Features")
plt.tight_layout()
plt.savefig("plots/correlation_heatmap.png", dpi=300, bbox_inches="tight")
plt.show()


# ---------------- MODEL TRAINING ----------------

log_model = LogisticRegression()
log_model.fit(X_train, y_train)
log_acc = accuracy_score(y_test, log_model.predict(X_test))

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
knn_acc = accuracy_score(y_test, knn.predict(X_test))

dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
dt_acc = accuracy_score(y_test, dt.predict(X_test))

svm = SVC(kernel='rbf', probability=True)
svm.fit(X_train, y_train)
svm_acc = accuracy_score(y_test, svm.predict(X_test))

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
rf_acc = accuracy_score(y_test, y_pred_rf)


# ---------------- CONFUSION MATRIX ----------------

cm = confusion_matrix(y_test, y_pred_rf)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix - Random Forest")
plt.tight_layout()
plt.savefig("plots/confusion_matrix_rf.png", dpi=300, bbox_inches="tight")
plt.show()


# ---------------- CLASSIFICATION REPORT ----------------

print(classification_report(y_test, y_pred_rf))


# ---------------- ROC CURVE ----------------

y_prob = rf.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label="ROC Curve (AUC = %0.2f)" % roc_auc)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Random Forest")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("plots/roc_curve_rf.png", dpi=300, bbox_inches="tight")
plt.show()


# ---------------- K-MEANS CLUSTERING ----------------

kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

plt.figure()
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, s=10)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("K-Means Clustering of Exoplanets")
plt.tight_layout()
plt.savefig("plots/kmeans_clustering.png", dpi=300, bbox_inches="tight")
plt.show()


# ---------------- RESULTS SUMMARY ----------------

results = pd.DataFrame({
    'Model': [
        'Logistic Regression',
        'KNN',
        'Decision Tree',
        'SVM',
        'Random Forest'
    ],
    'Accuracy': [
        log_acc,
        knn_acc,
        dt_acc,
        svm_acc,
        rf_acc
    ]
})

print(results)

plt.figure()
plt.bar(results['Model'], results['Accuracy'])
plt.xticks(rotation=30)
plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison")
plt.tight_layout()
plt.savefig("plots/model_accuracy.png", dpi=300, bbox_inches="tight")
plt.show()
