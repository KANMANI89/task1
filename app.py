from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# ==========================================
# 1. மாடலை தயார் செய்தல் (ML Model Training)
# ==========================================
np.random.seed(42)
n_samples = 1000
data = {
    'age': np.random.randint(21, 65, size=n_samples),
    'annual_income': np.random.randint(25000, 150000, size=n_samples),
    'loan_amount': np.random.randint(5000, 50000, size=n_samples),
    'credit_score': np.random.randint(300, 850, size=n_samples),
    'employment_duration_years': np.random.randint(0, 20, size=n_samples),
    'existing_debts': np.random.randint(0, 30000, size=n_samples),
    'home_ownership': np.random.choice(['RENT', 'OWN', 'MORTGAGE'], size=n_samples)
}
df = pd.DataFrame(data)
score_formula = (df['annual_income'] * 0.3) + (df['credit_score'] * 0.5) - (df['loan_amount'] * 0.2) - (df['existing_debts'] * 0.1)
df['credit_worthy'] = np.where(score_formula > score_formula.median(), 1, 0)
df['debt_to_income_ratio'] = (df['existing_debts'] + df['loan_amount']) / df['annual_income']

X = df.drop(columns=['credit_worthy'])
y = df['credit_worthy']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

numerical_features = ['age', 'annual_income', 'loan_amount', 'credit_score', 'employment_duration_years', 'existing_debts', 'debt_to_income_ratio']
categorical_features = ['home_ownership']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
    ])

model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])
model_pipeline.fit(X_train, y_train)
print("Backend Server: ML Model trained and ready to serve!")

# ==========================================
# 2. Web Routings & Endpoints
# ==========================================
@app.route('/')
def home():
    # templates ஃபோல்டரில் உள்ள index.html பக்கத்தை காண்பிக்கும்
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.get_json()
        input_df = pd.DataFrame([input_data])
        
        # Feature engineering
        input_df['debt_to_income_ratio'] = (input_df['existing_debts'] + input_df['loan_amount']) / input_df['annual_income']
        
        prediction = model_pipeline.predict(input_df)
        probability = model_pipeline.predict_proba(input_df)[:, 1][0]
        
        status = "Approved (Eligible for Loan)" if prediction[0] == 1 else "Rejected (Not Eligible)"
        
        return jsonify({
            'status': 'success',
            'loan_status': status,
            'approval_probability': f"{probability * 100:.2f}%"
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
