import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load your dataset
df = pd.read_csv("UCI Heart Disease Dataset.csv")

# Select features and target (update these columns as needed)
X = df[["age", "sex", "chol", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal" ]]  # Input features
y = df["target"]  # Target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Train a Random Forest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save the model as a pickle file
model_path = "Random_forest_model.pkl"
with open(model_path, "wb") as model_file:
    pickle.dump(model, model_file)

print(f"Model saved to {model_path}")
