import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

df = pd.read_csv("Churn_Modelling.csv")

X = df.drop(['RowNumber','CustomerId','Surname','Exited'], axis=1)
X = pd.get_dummies(X, drop_first=True)
y = df['Exited']

model = RandomForestClassifier()
model.fit(X, y)

joblib.dump(model, "model.pkl")

print("model.pkl created!")
