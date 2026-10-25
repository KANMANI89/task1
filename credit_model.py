import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

# =====================================================================
# 1. மாதிரி தரவுத்தொகுப்பை உருவாக்குதல் (Generating Sample Dataset)
# =====================================================================
print("Generating synthetic credit data...")
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

# நபர் கடன் செலுத்த தகுதியானவரா (1) இல்லையா (0) என்பதை தீர்மானிக்கும் லாஜிக் (Target Variable)
# அதிக வருமானம் மற்றும் அதிக கிரெடிட் ஸ்கோர் உள்ளவர்களுக்கு கடன் ஒப்புதல் (1) கிடைக்கும்
score_formula = (df['annual_income'] * 0.3) + (df['credit_score'] * 0.5) - (df['loan_amount'] * 0.2) - (df['existing_debts'] * 0.1)
df['credit_worthy'] = np.where(score_formula > score_formula.median(), 1, 0)

# =====================================================================
# 2. அம்ச பொறியியல் (Feature Engineering)
# =====================================================================
print("Performing feature engineering...")
# வருமானத்திற்கும் கடனுக்குமான விகிதத்தை புதிய அம்சமாக உருவாக்குதல் (Debt-to-Income Ratio)
df['debt_to_income_ratio'] = (df['existing_debts'] + df['loan_amount']) / df['annual_income']

# =====================================================================
# 3. தரவுப் பிரிப்பு (Data Splitting)
# =====================================================================
X = df.drop(columns=['credit_worthy'])
y = df['credit_worthy']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# =====================================================================
# 4. தரவு செயலாக்க பைப்லைன் (Data Preprocessing Pipeline)
# =====================================================================
numerical_features = ['age', 'annual_income', 'loan_amount', 'credit_score', 'employment_duration_years', 'existing_debts', 'debt_to_income_ratio']
categorical_features = ['home_ownership']

# எண்களை சீரமைக்கவும் (Scaling), உரை விவரங்களை எண்களாக மாற்றவும் (One-Hot Encoding) பைப்லைன்
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ])

# =====================================================================
# 5. மாடல் பயிற்சி (Model Training - Random Forest)
# =====================================================================
print("Training the Random Forest Classifier...")
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

model_pipeline.fit(X_train, y_train)
print("Model training completed successfully!\n")

# =====================================================================
# 6. மாடல் மதிப்பீடு (Model Evaluation Metrics)
# =====================================================================
y_pred = model_pipeline.predict(X_test)
y_pred_proba = model_pipeline.predict_proba(X_test)[:, 1]

print("="*50)
print("CLASSIFICATION REPORT:")
print("="*50)
print(classification_report(y_test, y_pred))

roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"ROC-AUC Score: {roc_auc:.4f}\n")

# =====================================================================
# 7. வரைபடங்கள் மற்றும் முடிவுகள் (Visualizations)
# =====================================================================
# Confusion Matrix வரைபடம்
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non Worthy', 'Worthy'], yticklabels=['Non Worthy', 'Worthy'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

# ROC Curve வரைபடம்
plt.subplot(1, 2, 2)
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (Area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")

plt.tight_layout()
print("Saving evaluation plots as 'credit_model_evaluation.png'...")
plt.savefig('credit_model_evaluation.png')
plt.show()