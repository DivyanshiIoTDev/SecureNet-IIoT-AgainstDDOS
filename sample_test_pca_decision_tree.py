import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pickle
import time

# Load cleaned data
df_cleaned = pd.read_parquet("train_preprocessed.parquet", engine="fastparquet")
X = df_cleaned.drop('Label', axis=1)  # Features (adjust 'Label' to your target column)
y = df_cleaned['Label']  # Target variable

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================
# METHOD 1: DECISION TREE FEATURE IMPORTANCE
# ============================================================
print("="*60)
print("METHOD 1: DECISION TREE FEATURE IMPORTANCE")
print("="*60)

start_time = time.time()

# Train Decision Tree
dt = DecisionTreeClassifier(random_state=42, max_depth=20)
dt.fit(X_train_scaled, y_train)

# Get feature importance
feature_importance_dt = pd.DataFrame({
    'feature': X.columns,
    'importance': dt.feature_importances_
}).sort_values('importance', ascending=False)

# Select top features (e.g., top 20 or features with importance > threshold)
IMPORTANCE_THRESHOLD_DT = 0.001  # Adjust this value
selected_features_dt = feature_importance_dt[
    feature_importance_dt['importance'] > IMPORTANCE_THRESHOLD_DT
]['feature'].tolist()

print(f"\nNumber of features selected: {len(selected_features_dt)} out of {len(X.columns)}")
print(f"Top 10 important features:\n{feature_importance_dt.head(10)}")

# Prepare data with selected features
X_train_dt = X_train_scaled[:, [X.columns.get_loc(f) for f in selected_features_dt]]
X_test_dt = X_test_scaled[:, [X.columns.get_loc(f) for f in selected_features_dt]]

dt_time = time.time() - start_time

# ============================================================
# METHOD 2: PCA (PRINCIPAL COMPONENT ANALYSIS)
# ============================================================
print("\n" + "="*60)
print("METHOD 2: PCA (PRINCIPAL COMPONENT ANALYSIS)")
print("="*60)

start_time = time.time()

# Apply PCA
# Option 1: Keep 95% of variance
pca = PCA(n_components=0.95, random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print(f"\nNumber of components: {pca.n_components_} out of {len(X.columns)}")
print(f"Explained variance ratio: {pca.explained_variance_ratio_[:10]}")
print(f"Cumulative explained variance: {np.cumsum(pca.explained_variance_ratio_)[-1]:.4f}")

pca_time = time.time() - start_time

# ============================================================
# EVALUATION: Compare Both Methods
# ============================================================
print("\n" + "="*60)
print("COMPARISON: EVALUATING BOTH METHODS")
print("="*60)

# Use a simple classifier to evaluate feature selection quality
# (We'll use Random Forest as it's fast and reliable)

results = {}

# --------- Test 1: Decision Tree Features ---------
print("\n[1] Evaluating with Decision Tree Selected Features:")
print(f"    Number of features: {X_train_dt.shape[1]}")

rf_dt = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_dt.fit(X_train_dt, y_train)

y_pred_dt = rf_dt.predict(X_test_dt)
acc_dt = accuracy_score(y_test, y_pred_dt)
prec_dt = precision_score(y_test, y_pred_dt, average='weighted', zero_division=0)
rec_dt = recall_score(y_test, y_pred_dt, average='weighted', zero_division=0)
f1_dt = f1_score(y_test, y_pred_dt, average='weighted', zero_division=0)

# Cross-validation
cv_scores_dt = cross_val_score(rf_dt, X_train_dt, y_train, cv=5, scoring='accuracy')

results['Decision Tree'] = {
    'accuracy': acc_dt,
    'precision': prec_dt,
    'recall': rec_dt,
    'f1_score': f1_dt,
    'cv_mean': cv_scores_dt.mean(),
    'cv_std': cv_scores_dt.std(),
    'n_features': X_train_dt.shape[1],
    'time': dt_time
}

print(f"    Accuracy:  {acc_dt:.4f}")
print(f"    Precision: {prec_dt:.4f}")
print(f"    Recall:    {rec_dt:.4f}")
print(f"    F1-Score:  {f1_dt:.4f}")
print(f"    CV Score (mean ± std): {cv_scores_dt.mean():.4f} ± {cv_scores_dt.std():.4f}")
print(f"    Processing time: {dt_time:.4f}s")

# --------- Test 2: PCA Features ---------
print("\n[2] Evaluating with PCA Features:")
print(f"    Number of components: {X_train_pca.shape[1]}")

rf_pca = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_pca.fit(X_train_pca, y_train)

y_pred_pca = rf_pca.predict(X_test_pca)
acc_pca = accuracy_score(y_test, y_pred_pca)
prec_pca = precision_score(y_test, y_pred_pca, average='weighted', zero_division=0)
rec_pca = recall_score(y_test, y_pred_pca, average='weighted', zero_division=0)
f1_pca = f1_score(y_test, y_pred_pca, average='weighted', zero_division=0)

# Cross-validation
cv_scores_pca = cross_val_score(rf_pca, X_train_pca, y_train, cv=5, scoring='accuracy')

results['PCA'] = {
    'accuracy': acc_pca,
    'precision': prec_pca,
    'recall': rec_pca,
    'f1_score': f1_pca,
    'cv_mean': cv_scores_pca.mean(),
    'cv_std': cv_scores_pca.std(),
    'n_features': X_train_pca.shape[1],
    'time': pca_time
}

print(f"    Accuracy:  {acc_pca:.4f}")
print(f"    Precision: {prec_pca:.4f}")
print(f"    Recall:    {rec_pca:.4f}")
print(f"    F1-Score:  {f1_pca:.4f}")
print(f"    CV Score (mean ± std): {cv_scores_pca.mean():.4f} ± {cv_scores_pca.std():.4f}")
print(f"    Processing time: {pca_time:.4f}s")

# ============================================================
# SUMMARY COMPARISON TABLE
# ============================================================
print("\n" + "="*60)
print("SUMMARY COMPARISON")
print("="*60)

comparison_df = pd.DataFrame(results).T
print("\n", comparison_df.round(4))

# ============================================================
# VISUALIZATION
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Accuracy Comparison
ax1 = axes[0, 0]
methods = list(results.keys())
accuracies = [results[m]['accuracy'] for m in methods]
colors = ['#2ecc71' if acc == max(accuracies) else '#3498db' for acc in accuracies]
ax1.bar(methods, accuracies, color=colors, alpha=0.7, edgecolor='black')
ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
ax1.set_title('Accuracy Comparison', fontsize=13, fontweight='bold')
ax1.set_ylim([min(accuracies) * 0.95, 1.0])
for i, v in enumerate(accuracies):
    ax1.text(i, v + 0.005, f'{v:.4f}', ha='center', fontweight='bold')

# Plot 2: F1-Score Comparison
ax2 = axes[0, 1]
f1_scores = [results[m]['f1_score'] for m in methods]
colors = ['#2ecc71' if f1 == max(f1_scores) else '#3498db' for f1 in f1_scores]
ax2.bar(methods, f1_scores, color=colors, alpha=0.7, edgecolor='black')
ax2.set_ylabel('F1-Score', fontsize=12, fontweight='bold')
ax2.set_title('F1-Score Comparison', fontsize=13, fontweight='bold')
ax2.set_ylim([min(f1_scores) * 0.95, 1.0])
for i, v in enumerate(f1_scores):
    ax2.text(i, v + 0.005, f'{v:.4f}', ha='center', fontweight='bold')

# Plot 3: Number of Features
ax3 = axes[1, 0]
n_features = [results[m]['n_features'] for m in methods]
colors = ['#e74c3c' if nf == max(n_features) else '#3498db' for nf in n_features]
ax3.bar(methods, n_features, color=colors, alpha=0.7, edgecolor='black')
ax3.set_ylabel('Number of Features', fontsize=12, fontweight='bold')
ax3.set_title('Dimensionality Comparison', fontsize=13, fontweight='bold')
for i, v in enumerate(n_features):
    ax3.text(i, v + 0.5, str(v), ha='center', fontweight='bold')

# Plot 4: Processing Time vs Accuracy Trade-off
ax4 = axes[1, 1]
times = [results[m]['time'] for m in methods]
accs = [results[m]['accuracy'] for m in methods]
ax4.scatter(times, accs, s=300, alpha=0.7, c=['#3498db', '#e74c3c'], edgecolors='black', linewidth=2)
for i, method in enumerate(methods):
    ax4.annotate(method, (times[i], accs[i]), xytext=(10, 10), 
                textcoords='offset points', fontweight='bold', fontsize=11)
ax4.set_xlabel('Processing Time (seconds)', fontsize=12, fontweight='bold')
ax4.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
ax4.set_title('Speed vs Accuracy Trade-off', fontsize=13, fontweight='bold')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('feature_selection_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n✓ Visualization saved as 'feature_selection_comparison.png'")

# ============================================================
# DECISION: Which Method is Better?
# ============================================================
print("\n" + "="*60)
print("RECOMMENDATION")
print("="*60)

best_method = max(results.keys(), key=lambda x: results[x]['f1_score'])
best_f1 = results[best_method]['f1_score']

print(f"\n🏆 BEST METHOD: {best_method}")
print(f"   F1-Score: {best_f1:.4f}")
print(f"\nKey Metrics Comparison:")
print(f"  • Decision Tree: {len(selected_features_dt)} features, {results['Decision Tree']['accuracy']:.4f} accuracy")
print(f"  • PCA: {X_train_pca.shape[1]} components, {results['PCA']['accuracy']:.4f} accuracy")

# Additional insight
if results['Decision Tree']['n_features'] < results['PCA']['n_features']:
    print(f"\n✓ Decision Tree is more interpretable ({results['Decision Tree']['n_features']} vs {results['PCA']['n_features']} features)")
else:
    print(f"\n✓ PCA reduces dimensionality more effectively ({results['PCA']['n_features']} vs {results['Decision Tree']['n_features']} features)")

# ============================================================
# SAVE RESULTS
# ============================================================

# Save the winning method
if best_method == 'Decision Tree':
    pickle.dump(selected_features_dt, open('04_selected_features.pkl', 'wb'))
    print(f"\n✓ Selected features saved to '04_selected_features.pkl'")
else:
    pickle.dump(pca, open('04_pca_model.pkl', 'wb'))
    print(f"\n✓ PCA model saved to '04_pca_model.pkl'")

# Save comparison results
comparison_df.to_csv('feature_selection_results.csv')
print("✓ Detailed results saved to 'feature_selection_results.csv'")