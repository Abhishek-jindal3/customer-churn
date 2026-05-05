import os
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report,confusion_matrix
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

df=pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df=df[df['TotalCharges'].notnull()]
df=df.drop('customerID',axis=1)


scaler = MinMaxScaler()
df[['TotalCharges','MonthlyCharges','tenure']] = scaler.fit_transform(df[['TotalCharges','MonthlyCharges','tenure']])
df.replace({'No phone service': 'No', 'No internet service': 'No'},inplace=True)
df.replace({'Male': 0, 'Female': 1, 'Yes': 1, 'No': 0}, inplace=True)
df = pd.get_dummies(df, columns=['InternetService','Contract','PaymentMethod'], drop_first=True)

X = df.drop('Churn', axis=1)
print(X.shape)
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape)

model=keras.Sequential([
    keras.layers.Input(shape=(23,)),
    keras.layers.Dense(16, input_shape=(23,), activation='relu'),
    keras.layers.Dense(8, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=20)
y_pred = (model.predict(X_test) > 0.5).astype(int)

print(classification_report(y_test, y_pred))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
