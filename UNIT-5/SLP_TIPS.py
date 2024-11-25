import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load Tips dataset
tips_url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv"
tips = pd.read_csv(tips_url)

# Preprocessing
tips['sex'] = tips['sex'].map({'Female': 0, 'Male': 1})
tips['smoker'] = tips['smoker'].map({'No': 0, 'Yes': 1})
tips['day'] = tips['day'].map({'Thur': 0, 'Fri': 1, 'Sat': 2, 'Sun': 3})
tips['time'] = tips['time'].map({'Lunch': 0, 'Dinner': 1})
tips = tips.dropna()  # Handle missing values

X = tips[['total_bill', 'tip', 'sex', 'smoker', 'day', 'time', 'size']]
y = tips['smoker']

scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# SLP Classifier
slp = LogisticRegression()
slp.fit(X_train, y_train)

# Predictions
y_pred = slp.predict(X_test)

# Accuracy
print("SLP Model Accuracy (Tips Dataset):", accuracy_score(y_test, y_pred))
