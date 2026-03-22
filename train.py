import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import kagglehub
import os

print("=" * 60)
print("DISEASE PREDICTION MODEL TRAINING")
print("=" * 60)

# Download dataset from Kaggle
print("\n📥 Downloading dataset from Kaggle...")
try:
    dataset_path = kagglehub.dataset_download("kaushil268/disease-prediction-using-machine-learning")
    print(f"✅ Dataset downloaded to: {dataset_path}")
except Exception as e:
    print(f"⚠️ Could not download dataset: {e}")
    print("Please ensure Kaggle API is configured. Exiting...")
    exit(1)

# Find CSV files in the dataset
print("\n📂 Looking for CSV files...")
csv_files = []
for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.endswith('.csv'):
            csv_files.append(os.path.join(root, file))
            print(f"   Found: {file}")

if not csv_files:
    print("❌ No CSV files found in dataset!")
    exit(1)

# Load the first CSV file
data_file = csv_files[0]
print(f"\n📊 Loading data from: {os.path.basename(data_file)}")
df = pd.read_csv(data_file)

print(f"   Dataset shape: {df.shape}")
print(f"   Columns: {df.columns.tolist()}")
print(f"\n   First few rows:")
print(df.head())

# Data preprocessing
print("\n🔧 Preprocessing data...")

# Display basic info
print(f"   Missing values:\n{df.isnull().sum()}")

# Handle missing values
df = df.dropna()
print(f"   After removing NaN: {df.shape}")

# Identify target column (usually last column or common names like 'target', 'disease', 'outcome')
target_column = df.columns[-1]
print(f"\n   Target column detected: '{target_column}'")

# Separate features and target
X = df.drop(columns=[target_column])
y = df[target_column]

print(f"   Features: {X.shape[1]}")
print(f"   Samples: {X.shape[0]}")
print(f"   Target distribution:\n{y.value_counts()}")

# Handle categorical columns by encoding
print("\n🔤 Encoding categorical features...")
for col in X.select_dtypes(include=['object']).columns:
    X[col] = pd.factorize(X[col])[0]
    print(f"   Encoded: {col}")

# Remove rare symptoms (appearing in less than 10% of samples)
print("\n🔍 Removing rare symptoms...")
threshold = len(X) * 0.10  # 10% of samples
symptom_counts = (X == 1).sum()  # Count how many times each symptom appears
rare_symptoms = symptom_counts[symptom_counts < threshold].index.tolist()
X = X.drop(columns=rare_symptoms)
print(f"   Removed {len(rare_symptoms)} rare symptoms")
print(f"   Remaining features: {X.shape[1]} (from {X.shape[1] + len(rare_symptoms)})")
print(f"   Removed symptoms: {rare_symptoms}")

# Train-test split
print("\n📈 Splitting data (80-20)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"   Training set: {X_train.shape[0]} samples")
print(f"   Test set: {X_test.shape[0]} samples")

# Feature scaling
print("\n⚖️  Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("   ✅ Features scaled")

# Train the model
print("\n🤖 Training Random Forest model...")
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train_scaled, y_train)
print("   ✅ Model training complete")

# Make predictions
print("\n🔮 Making predictions on test set...")
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)

# Evaluate model
print("\n📊 MODEL EVALUATION")
print("=" * 60)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# Feature importance
print("\n🎯 Top 10 Important Features:")
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

for idx, row in feature_importance.head(10).iterrows():
    print(f"   {row['Feature']:30s}: {row['Importance']:.4f}")

# Save the model and scaler
print("\n💾 Saving model and scaler...")
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Save feature names for later use
with open('feature_names.pkl', 'wb') as f:
    pickle.dump(X.columns.tolist(), f)

print("   ✅ model.pkl saved")
print("   ✅ scaler.pkl saved")
print("   ✅ feature_names.pkl saved")

print("\n" + "=" * 60)
print("✅ TRAINING COMPLETE")
print("=" * 60)
