import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# --- 1. LOAD DATA ---
if not os.path.exists('my_data.csv'):
    print("Error: 'my_data.csv' not found. Record some data first!")
    exit()

df = pd.read_csv('my_data.csv', header=None)

# Basic check: Ensure we have the new 126-feature columns (plus 1 label column)
if df.shape[1] < 127:
    print(f"Warning: Found {df.shape[1]-1} features. Are you sure you recorded two-hand data?")

X = df.iloc[:, 1:].values # Landmarks (126 features)
y = df.iloc[:, 0].values  # Word Labels

# --- 2. SPLIT ---
# Using 20% for testing to ensure the model generalizes well
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 3. TRAIN ---
print(f"🚀 Training Two-Hand Model on {len(X)} samples...")
# Increased n_estimators for better handling of the larger feature set
model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1) 
model.fit(X_train, y_train)

# --- 4. STATS ---
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("-" * 30)
print(f"✅ Model Accuracy: {acc*100:.2f}%")
print("-" * 30)
print("\nWord Breakdown:\n", classification_report(y_test, y_pred))

# --- 5. SAVE ---
joblib.dump(model, 'asl_model.pkl')
print("\n💾 Model saved as 'asl_model.pkl'")