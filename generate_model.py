import joblib
import numpy as np
import sys
import traceback

def create_simple_model():
    try:
        # Try to import scikit-learn
        from sklearn.ensemble import RandomForestClassifier
        
        # Create a simple Random Forest model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Create some dummy data for training
        np.random.seed(42)
        X = np.random.rand(100, 4) * 100  # 4 features: attendance, score, attempts, fees
        y = (X[:, 0] < 50) | (X[:, 1] < 40) | (X[:, 3] > 500)  # Simple rule for dropout risk
        
        # Train the model
        model.fit(X, y)
        
        # Save the model
        joblib.dump(model, 'rf_model.joblib')
        print("✅ Model created and saved successfully!")
        return True
        
    except ImportError as e:
        print(f"❌ Error importing scikit-learn: {e}")
        print("Please install scikit-learn: pip install scikit-learn")
        return False
    except Exception as e:
        print(f"❌ Error creating model: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = create_simple_model()
    if success:
        print("You can now run the app with: streamlit run app.py")
    else:
        print("Failed to create model. Please check the error messages above.")
