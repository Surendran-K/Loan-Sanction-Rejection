import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('loan_data.csv')

# Preprocess data
le = LabelEncoder()
for i in df.select_dtypes(include='object').columns:
    df[i] = le.fit_transform(df[i])

for i in df.columns[:-1]:
    q1 = df[i].quantile(0.25)
    q3 = df[i].quantile(0.75)
    iqr = q3 - q1
    ll = q1 - 1.5 * iqr
    ul = q3 + 1.5 * iqr
    df[i] = np.where(df[i] > ul, ul, np.where(df[i] < ll, ll, df[i]))

rs = RobustScaler()
x = df.drop('loan_status', axis=1)
y = df['loan_status']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y)
x_train_sc = pd.DataFrame(data=rs.fit_transform(x_train), columns=x_train.columns)
x_test_sc = pd.DataFrame(data=rs.transform(x_test), columns=x_test.columns)

# Define models
models = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'KNeighbours': KNeighborsClassifier(),
    'Naive Bayes': BernoulliNB(),
    'XGBoost': XGBClassifier()
}

# Train models
for name, model in models.items():
    model.fit(x_train_sc, y_train)

# Streamlit interface
st.title('Loan Approval / Rejection Model')

# User input form with instructions
st.sidebar.header('User Input Features')

# Instructions for users
st.sidebar.write("**Instructions:**")
st.sidebar.write("**Numerical Input:** Please provide numerical values. Example: Age (e.g., 25), Loan Amount (e.g., 5000)")
st.sidebar.write("**Categorical Input (Gender):**  1 for male, 0 for female" )
st.sidebar.write("**Categorical Input (Education):** 0 - Associate, 1 -Bachelors, 2 - Doctorate , 3 -High School, 4 - Masters")
st.sidebar.write("**Categorical Input (Home Ownership):** 0 - Mortgage, 1 - Other, 2 - Own, 3- Rent")
st.sidebar.write("**Categorical Input (Loan Intent):** DEBT CONSOLIDATION -0 , Education -1, Home Improvement -2, Medical -3, Personal -4,  Venture - 5")
st.sidebar.write("**Categorical Input(Previous Default):** 0 - No, 1 - Yes")

def user_input_features():
    inputs = {}
    for col in x.columns:
        if df[col].dtype == 'object' or df[col].dtype == 'category':
            inputs[col] = st.sidebar.number_input(col, min_value=0.0)
        else:
            inputs[col] = st.sidebar.number_input(col, value=0)
    return pd.DataFrame(inputs, index=[0])

user_input = user_input_features()
user_input_scaled = rs.transform(user_input)
predictions = {name: model.predict(user_input_scaled)[0] for name, model in models.items()}

st.subheader('User Input:')
st.write(user_input)

# Display predictions
st.subheader('Model Predictions:')
for name, prediction in predictions.items():
    st.write(f"{name}: {'Approved' if prediction == 1 else 'Rejected'}")

# Plotting accuracy
st.subheader('Model Performance')

train_accuracies = {name: accuracy_score(y_train, model.predict(x_train_sc)) for name, model in models.items()}
test_accuracies = {name: accuracy_score(y_test, model.predict(x_test_sc)) for name, model in models.items()}

st.subheader('Train Accuracy')
fig, ax1 = plt.subplots(figsize=(9, 6))
sns.barplot(x=list(train_accuracies.keys()), y=list(train_accuracies.values()), ax=ax1)
for container in ax1.containers:
    labels = [f'{x:.2f}' for x in container.datavalues * 100]
    ax1.bar_label(container, labels=labels)
st.pyplot(fig)

st.subheader('Test Accuracy')
fig, ax = plt.subplots(figsize=(9, 6))
sns.barplot(x=list(test_accuracies.keys()), y=list(test_accuracies.values()), ax=ax)
for container in ax.containers:
    labels = [f'{x:.2f}' for x in container.datavalues * 100]
    ax.bar_label(container, labels=labels)
st.pyplot(fig)
