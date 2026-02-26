import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os

class TumorClassifier:

    def __init__(self):
        # Random Forest — ensemble of decision trees
        # n_estimators=100 means 100 trees vote on the result
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)

    def train(self, features, labels):
        # Split data — 80% training, 20% testing
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42
        )

        # Train the model
        self.model.fit(X_train, y_train)

        # Test accuracy
        predictions = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
        print("\nDetailed Report:")
        print(classification_report(y_test, predictions, target_names=["Normal", "Tumor"]))

        return accuracy

    def predict(self, features):
        # Predict single image
        prediction = self.model.predict([features])[0]
        probability = self.model.predict_proba([features])[0]
        label = "Tumor" if prediction == 1 else "Normal"
        confidence = max(probability) * 100
        return label, confidence

    def save(self, path="../../output/model.pkl"):
        # Save trained model to file
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"Model saved to {path}")

    def load(self, path="../../output/model.pkl"):
        # Load previously trained model
        with open(path, 'rb') as f:
            self.model = pickle.load(f)