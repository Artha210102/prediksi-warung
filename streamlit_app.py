import streamlit as st
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Set page title
st.title("House Price Prediction using SVR")

# Load data
@st.cache
def load_data():
    return pd.read_csv("kc_house_data.csv")

df = load_data()

# Sidebar for navigation
st.sidebar.header("Navigation")
options = st.sidebar.radio("Go to:", ["Data Exploration", "Model Training", "Prediction"])

# Data Exploration
if options == "Data Exploration":
    st.header("Exploratory Data Analysis")

    st.subheader("Dataset Overview")
    st.write(df.head())

    st.subheader("Statistical Summary")
    st.write(df.describe())

    st.subheader("Missing Values")
    st.write(df.isnull().sum())

    st.subheader("Univariate Analysis")
    feature = st.selectbox("Select feature for analysis", df.columns)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(df[feature], kde=True, ax=ax[0])
    sns.boxplot(x=df[feature], ax=ax[1])
    st.pyplot(fig)

    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# Model Training
elif options == "Model Training":
    st.header("Train the SVR Model")

    # Select features
    features = st.multiselect("Select features for prediction", 
                               df.columns[:-1], default=['bedrooms', 'bathrooms', 'sqft_living', 'grade', 'yr_built'])

    x = df[features]
    y = df['price']

    # Train-test split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4)

    # Standardize data
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # Train SVR model
    svr = SVR(kernel='rbf', C=100, epsilon=0.1)
    svr.fit(x_train_scaled, y_train)

    # Evaluate model
    y_pred = svr.predict(x_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.write("**Model Performance:**")
    st.write(f"Mean Squared Error: {mse:.2f}")
    st.write(f"R² Score: {r2:.2f}")

# Prediction
elif options == "Prediction":
    st.header("Predict House Price")

    st.write("Enter the features of the house:")

    # Filter only numeric columns for user input
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns

    input_features = []
    for col in numeric_columns:
        val = st.number_input(f"{col}", value=float(df[col].mean()))
        input_features.append(val)

    input_features = [input_features]
    scaler = StandardScaler()
    scaler.fit(df[numeric_columns])  # Fit the scaler on the numeric columns
    input_scaled = scaler.transform(input_features)

    # Train SVR model
    svr = SVR(kernel='rbf', C=100, epsilon=0.1)
    svr.fit(scaler.fit_transform(df[numeric_columns]), df['price'])  # Fit the model on the numeric columns
    predicted_price = svr.predict(input_scaled)

    st.write(f"**Predicted Price:** ${predicted_price[0]:,.2f}")
