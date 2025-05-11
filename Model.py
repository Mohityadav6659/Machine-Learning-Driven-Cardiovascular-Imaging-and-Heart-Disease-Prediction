import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import joblib

# 1. Load raw data
df = pd.read_csv(r'C:\Users\Public\OneDrive\Desktop\ML Code\hearts.csv')

# 2. Encode categoricals in-place
df['Gender']         = df['Gender'].map({'M': 0, 'F': 1})
df['ChestPainType']  = df['ChestPainType'].map({'ATA': 0, 'NAP': 1, 'ASY': 2, 'TA': 3})
df['RestingECG']     = df['RestingECG'].map({'Normal': 0, 'ST': 1})
df['ExerciseAngina'] = df['ExerciseAngina'].map({'N': 0, 'Y': 1})
df['ST_Slope']       = df['ST_Slope'].map({'Up': 0, 'Flat': 1, 'Down': 2})

# 3. Split off features (X) and target (y)
X = df.drop(columns=['HeartDisease'])
y = df['HeartDisease']

# 4. Fit imputer on X only, then transform X
imputer = SimpleImputer(strategy='mean')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# 5. Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# 6. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# 7. Choose and train model
print("\nChoose a model:")
print("1) Logistic Regression")
print("2) Random Forest")
print("3) Support Vector Machine")
choice = input("Enter 1, 2 or 3: ").strip()

if choice == '1':
    model = LogisticRegression(max_iter=1000); name = "Logistic Regression"
elif choice == '2':
    model = RandomForestClassifier();      name = "Random Forest"
elif choice == '3':
    model = SVC(probability=True);         name = "Support Vector Machine"
else:
    print("Invalid → defaulting to Logistic Regression")
    model = LogisticRegression(max_iter=1000); name = "Logistic Regression"

model.fit(X_train, y_train)

# 8. Evaluate & plot ROC
acc  = model.score(X_test, y_test)
pred = model.predict(X_test)
cm   = confusion_matrix(y_test, pred)
fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:,1])
roc_auc     = auc(fpr, tpr)

print(f"\n{name} Accuracy: {acc:.3f}")
print("\nClassification Report:\n", classification_report(y_test, pred))
print("\nConfusion Matrix:\n", cm)

plt.figure(figsize=(5,4))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0,1],[0,1],'--',color='grey')
plt.title(f"{name} ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()

# ── Additional Plots ──

# Confusion Matrix Heatmap
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No','Yes'], yticklabels=['No','Yes'])
plt.title(f'{name} Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Feature Importances (if available)
if hasattr(model, 'feature_importances_'):
    importances = model.feature_importances_
    features = X.columns
    plt.figure(figsize=(8,6))
    plt.barh(features, importances)
    plt.title(f'Feature Importances for {name}')
    plt.xlabel('Importance')
    plt.show()

# Accuracy Score Bar
plt.figure(figsize=(4,3))
plt.bar([name], [acc], color='seagreen')
plt.title('Model Accuracy')
plt.ylim(0,1)
plt.ylabel('Accuracy')
plt.show()

# 9. Save artifacts
joblib.dump(model,   "model.pkl")
joblib.dump(imputer, "imputer.pkl")
joblib.dump(scaler,  "scaler.pkl")
print("\nSaved model.pkl, imputer.pkl, scaler.pkl")

# 10. — Now prompt user for new patient data —
print("\nEnter patient details to predict heart-disease risk:")

ui = {}
ui['Age']            = float(input("Age: "))
ui['Gender']         = input("Gender (M/F): ")
ui['ChestPainType']  = input("ChestPainType (ATA/NAP/ASY/TA): ")
ui['RestingBP']      = float(input("RestingBP: "))
ui['Cholesterol']    = float(input("Cholesterol: "))
ui['FastingBS']      = float(input("FastingBS (0 or 1): "))
ui['RestingECG']     = input("RestingECG (Normal/ST): ")
ui['MaxHR']          = float(input("MaxHR: "))
ui['ExerciseAngina'] = input("ExerciseAngina (Y/N): ")
ui['Oldpeak']        = float(input("Oldpeak: "))
ui['ST_Slope']       = input("ST_Slope (Up/Flat/Down): ")

# Map them exactly as training
user_df = pd.DataFrame([ui])
user_df['Gender']         = user_df['Gender'].map({'M':0,'F':1})
user_df['ChestPainType']  = user_df['ChestPainType'].map({'ATA':0,'NAP':1,'ASY':2,'TA':3})
user_df['RestingECG']     = user_df['RestingECG'].map({'Normal':0,'ST':1})
user_df['ExerciseAngina'] = user_df['ExerciseAngina'].map({'N':0,'Y':1})
user_df['ST_Slope']       = user_df['ST_Slope'].map({'Up':0,'Flat':1,'Down':2})

# Impute & scale
user_imputed = imputer.transform(user_df)    # now matches X.columns
user_scaled  = scaler.transform(user_imputed)

# Predict
risk = model.predict(user_scaled)[0]

print("\n>>> Prediction:")
if risk == 1:
    print("High risk of heart disease 💔")
else:
    print("Low risk of heart disease ❤️")

# 11. Plot a single-bar so you can see it even at 0
plt.figure(figsize=(4,3))
plt.bar(['Risk'], [risk], color=('salmon' if risk else 'lightgreen'))
plt.ylim(-0.1, 1.1)
plt.ylabel('1 = High risk, 0 = Low risk')
plt.show()
