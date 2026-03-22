from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model, scaler, and feature names
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('feature_names.pkl', 'rb') as f:
    FEATURE_NAMES = pickle.load(f)

# Get disease classes from the trained model
DISEASE_CLASSES = model.classes_.tolist()

@app.route('/')
def home():
    """Home page"""
    return render_template('home.html', page='home')

@app.route('/predict')
def predict_page():
    """Predict page"""
    return render_template('predict.html', page='predict')

@app.route('/history')
def history():
    """History page"""
    return render_template('history.html', page='history')

@app.route('/about')
def about():
    """About page"""
    return render_template('about.html', page='about')

@app.route('/predict', methods=['POST'])
def predict():
    """Make prediction based on input features"""
    try:
        # Get JSON data from request
        data = request.get_json()
        
        # Extract features in the correct order
        features = []
        for feature in FEATURE_NAMES:
            value = data.get(feature)
            if value is None:
                return jsonify({'error': f'{feature} is required'}), 400
            features.append(float(value))
        
        # Convert to numpy array
        features_array = np.array([features])
        
        # Scale features
        scaled_features = scaler.transform(features_array)
        
        # Make prediction
        prediction_disease = model.predict(scaled_features)[0]
        probabilities = model.predict_proba(scaled_features)[0]
        
        # Get the index of the predicted disease
        prediction_idx = np.where(model.classes_ == prediction_disease)[0][0]
        
        # Find top 3 predictions with their probabilities
        top_indices = np.argsort(probabilities)[-3:][::-1]
        top_predictions = [
            {
                'disease': DISEASE_CLASSES[int(idx)],
                'probability': float(probabilities[int(idx)])
            }
            for idx in top_indices
        ]
        
        # Return results
        return jsonify({
            'predicted_disease': prediction_disease,
            'confidence': float(probabilities[prediction_idx]),
            'top_predictions': top_predictions
        })
    
    except ValueError as e:
        return jsonify({'error': f'Invalid input: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/features', methods=['GET'])
def get_features():
    """Return available features"""
    return jsonify({'features': FEATURE_NAMES})

if __name__ == '__main__':
    app.run(debug=True)
