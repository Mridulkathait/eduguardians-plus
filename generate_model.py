import joblib
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Create a simple Random Forest model
def create_simple_model():
    # Initialize the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Create some dummy data for training
    np.random.seed(42)
    X = np.random.rand(100, 4) * 100  # 4 features: attendance, score, attempts, fees
    y = (X[:, 0] < 50) | (X[:, 1] < 40) | (X[:, 3] > 500)  # Simple rule for dropout risk
    
    # Train the model
    model.fit(X, y)
    
    # Save the model
    joblib.dump(model, 'rf_model.joblib')
    print("Model created and saved successfully!")

if __name__ == "__main__":
    create_simple_model()
