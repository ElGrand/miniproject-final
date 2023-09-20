#!/usr/bin/env python
# coding: utf-8

#get_ipython().system('pip install streamlit')

# house_price_app.py

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns #Multiple Linear Regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics #Multiple Linear Regression
from sklearn.metrics import r2_score #Multiple Linear Regression & Polynomial
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import sklearn.metrics as sm
from sklearn.preprocessing import PolynomialFeatures # Polynomial Regression

# Read the data
df = pd.read_csv("./data/house-data.csv", sep=',')

# Set page title
st.title("House Price Prediction App")

# Display data exploration section
st.header("Data Exploration")
st.subheader("Data Shape")
st.write(f"Number of Rows: {df.shape[0]}, Number of Columns: {df.shape[1]}")

st.subheader("Column Names")
st.write(list(df.columns))

st.subheader("Data Info")
st.write(df.info())

st.subheader("Descriptive Statistics")
st.write(df.describe())

st.subheader("Sample Data")
st.write(df.sample())

st.subheader("Price vs Sqft Living")
plt.xlabel('sqft_living')
plt.ylabel('price')
plt.scatter(df.sqft_living, df.price, color='green')
st.pyplot()

st.subheader("Price Distribution")
sns.distplot(df['price'], label='price', norm_hist=True)
st.pyplot()

st.subheader("Sqft Living Distribution")
sns.distplot(df['sqft_living'], label='sqft_living', norm_hist=True)
st.pyplot()

# Data cleaning section
st.header("Data Cleaning")
st.subheader("Null Values")
st.write(df.isnull().sum())

st.subheader("Null Value Heatmap")
sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='viridis')
st.pyplot()

# Data dependencies section
st.header("Data Dependencies")
corr_matrix = df.corr()
st.write("Correlation Matrix")
st.write(corr_matrix)

plt.subplots(figsize=(18, 18))
sns.heatmap(corr_matrix, annot=True)
st.pyplot()

# Model Training section
st.header("Model Training")

X = df['sqft_living'].values.reshape(-1, 1)
y = df['price'].values.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=5000, test_size=0.15)

st.subheader("Training Data Split")
st.write(f"X_train shape: {X_train.shape}")
st.write(f"y_train shape: {y_train.shape}")
st.write(f"X_test shape: {X_test.shape}")
st.write(f"y_test shape: {y_test.shape}")

myreg = LinearRegression()
myreg.fit(X_train, y_train)

a = myreg.coef_
b = myreg.intercept_

st.subheader("Linear Regression Model")
st.write(f"Coefficient (a): {a}")
st.write(f"Intercept (b): {b}")

y_predicted = myreg.predict(X_test)

st.subheader("Linear Regression Model Visualization")
plt.title('Linear Regression')
plt.scatter(X, y, color='green')
plt.plot(X_train, a * X_train + b, color='blue')
plt.plot(X_test, y_predicted, color='orange')
plt.xlabel('sqft_living')
plt.ylabel('price')
st.pyplot()

R2 = myreg.score(X, y)

st.subheader("Model R-squared Score")
st.write(f"R-squared Score: {R2}")

# Model Evaluation section
st.header("Model Evaluation")

# MAE
mae = np.mean(np.abs(y_test - y_predicted))
st.write(f"Mean Absolute Error (MAE): {mae}")

# MSE
mse = np.mean((y_test - y_predicted) ** 2)
st.write(f"Mean Squared Error (MSE): {mse}")

# RMSE
rmse = np.sqrt(mse)
st.write(f"Root Mean Squared Error (RMSE): {rmse}")

# Explained Variance Score
eV = round(sm.explained_variance_score(y_test, y_predicted), 2)
st.write(f"Explained Variance Score: {eV}")

# R-squared Score
r2 = r2_score(y_test, y_predicted)
st.write(f"R-squared Score: {r2}")

# Store Model
model_file = './deploy/mypolifit_miniproject3-linearRegression.pkl'

with open(model_file, 'wb') as f:
    pickle.dump(myreg, f)

st.subheader("Model Stored")
st.write(f"The trained model has been stored in '{model_file}'")

# Predictions section
st.header("Predictions")

sqft_living_input = st.text_input("Enter sqft_living for prediction:")
if sqft_living_input:
    sqft_living_input = float(sqft_living_input)
    price_predicted = myreg.predict(np.array([[sqft_living_input]]))
    st.write(f"Predicted Price for {sqft_living_input} sqft_living: {price_predicted[0][0]:.2f}")

#                            #
# Multiple Linear Regression #
#                            #

# Load the data
data = pd.read_csv("./data/house-data.csv", sep=',')

# Page title
st.title("Multiple Linear Regression Web App")

# Data exploration
st.header("Data Exploration")
st.subheader("Data Summary")
st.write(data.describe())

# Visualize the features and response using scatterplots
st.subheader("Scatterplot")
sns.pairplot(data, x_vars=['sqft_living', 'grade', 'sqft_basement'], y_vars='price', height=5, aspect=0.8)
st.pyplot()

# Model training
st.header("Model Training")
feature_cols = ['sqft_living', 'grade', 'sqft_basement']
X = data[feature_cols]
y = data['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
linreg = LinearRegression()
linreg.fit(X_train, y_train)

# Model evaluation
st.header("Model Evaluation")
y_predicted = linreg.predict(X_test)
mae = metrics.mean_absolute_error(y_test, y_predicted)
mse = metrics.mean_squared_error(y_test, y_predicted)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_predicted)

st.write("Mean Absolute Error (MAE):", mae)
st.write("Mean Squared Error (MSE):", mse)
st.write("Root Mean Squared Error (RMSE):", rmse)
st.write("R-squared (R2):", r2)

# Regression results visualization
st.header("Regression Results Visualization")
plt.title('Multiple Linear Regression')
plt.scatter(y_test, y_predicted, color='blue')
st.pyplot()

# Model improvement
st.header("Model Improvement")
st.write("To improve the model, you can consider using more features.")

#                       #
# Polynomial Regression #
#                       #

# Load the data
df = pd.read_csv("./data/house-data.csv")

# Sidebar
st.sidebar.title("Regression Options")
regression_type = st.sidebar.radio("Select Regression Type", ("Linear Regression", "Polynomial Regression"))

# Main content
st.title("House Price Prediction")
st.markdown("Explore different regression models for house price prediction.")

# Scatter plot of the data
# Scatter plot of the data
st.subheader("Scatter Plot of Data")
fig, ax = plt.subplots()
ax.scatter(df['sqft_living'], df['price'], color='red')
st.pyplot(fig)
st.write("This scatter plot shows the relationship between sqft_living and price.")

# Regression model selection
st.subheader("Regression Model")

if regression_type == "Linear Regression":
    st.write("### Linear Regression")
    X = df['sqft_living'].values.reshape(-1, 1)
    y = df['price'].values.reshape(-1, 1)
    lin_reg = LinearRegression()
    lin_reg.fit(X, y)
    
    # Create the scatter plot using Matplotlib
    fig, ax = plt.subplots()
    ax.scatter(X, y, color='red')
    ax.plot(X, lin_reg.predict(X), color='blue')
    ax.set_title('Linear Regression')
    ax.set_xlabel('sqft_living')
    ax.set_ylabel('price')

    # Save the figure as an image
    plt.savefig("linear_regression_plot.png")

    # Display the saved image using Streamlit
    st.image("linear_regression_plot.png")

    st.write("R-squared (R^2) score for Linear Regression:", round(lin_reg.score(X, y), 2))


else:  # Polynomial Regression
    st.write("### Polynomial Regression")
    poly_degree = st.slider("Select Polynomial Degree", min_value=2, max_value=10, value=5)
    
    # Convert the 'sqft_living' column to a numeric data type
    X = df['sqft_living'].astype(float).values.reshape(-1, 1)
    
    poly_model = PolynomialFeatures(degree=poly_degree)
    X_poly = poly_model.fit_transform(X)
    pol_reg = LinearRegression()
    pol_reg.fit(X_poly, y)

    # Visualization of Polynomial Regression
    X_range = np.linspace(min(X), max(X), 100).reshape(-1, 1)
    
    fig, ax = plt.subplots()
    ax.scatter(X, y, color='red')
    ax.plot(X_range, pol_reg.predict(poly_model.transform(X_range)), color='blue')
    ax.set_title('Polynomial Regression')
    ax.set_xlabel('sqft_living')
    ax.set_ylabel('price')
    
    st.pyplot(fig)
    
    y_predict = pol_reg.predict(X_poly)
    st.write("R-squared (R^2) score for Polynomial Regression:", round(r2_score(y, y_predict), 2))


# Predictions
st.subheader("Make Predictions")
input_sqft_living = st.text_input("Enter sqft_living:")
try:
    input_sqft_living = float(input_sqft_living)  # Convert input to float
    if regression_type == "Linear Regression":
        prediction = lin_reg.predict([[input_sqft_living]])
    else:
        prediction = pol_reg.predict(poly_model.transform([[input_sqft_living]]))
    
    if prediction.size > 0:  # Check if prediction has a valid value
        st.write(f"Predicted Price: ${round(prediction[0], 2)}")
    else:
        st.write("Could not make a prediction.")
except ValueError:
    st.write("Please enter a valid numeric value for sqft_living.")



# About
st.sidebar.title("About")
st.sidebar.info(
    "This web app demonstrates Linear and Polynomial Regression models for house price prediction using Streamlit."
)
# Credits
st.sidebar.title("Credits")
st.sidebar.info("By Your Name Here")
# Run the app
if __name__ == "__main__":
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.set_option('deprecation.showfileUploaderEncoding', False)
    st.set_option('deprecation.showfileUploaderEncoding', False)
    st.set_option('deprecation.showfileUploaderEncoding', False)
    st.set_option('deprecation.showfileUploaderEncoding', False)
    st.set_option('deprecation.showfileUploaderEncoding', False)
    st.set_option('deprecation.showfileUploaderEncoding', False)
    st.set_option('deprecation.showfileUploaderEncoding', False)
    st.set_option('deprecation.showfileUploaderEncoding', False)
    st.set_option('deprecation.showfileUploaderEncoding', False)
    st.set_option('deprecation.showfileUploaderEncoding', False)
    st.set_option('deprecation.showfileUploaderEncoding', False)
    st.set_option('deprecation.showfileUploaderEncoding', False)
    st.set_option('deprecation.showfileUploaderEncoding', False)
    st.set_option('deprecation.showfileUploaderEncoding', False)
    st.set_option('deprecation.showfileUploaderEncoding', False)
    st.set_option('deprecation.showfileUploaderEncoding', False)
    st.set_option('deprecation.showfileUploaderEncoding', False)
    st.set_option('deprecation.showfileUploaderEncoding', False)
    st.set_option('deprecation.showfileUploaderEncoding', False)
    st.set_option('deprecation.showfileUploaderEncoding', False)
    st.set_option('deprecation.showfileUploaderEncoding', False)
    st.set_option('deprecation.showfileUploaderEncoding', False)
    st.set_option('deprecation.showfileUploaderEncoding', False)
    st.set_option('deprecation.showfileUploaderEncoding', False)
    st.set_option('deprecation.showfileUploaderEncoding', False)
    st.set_option('deprecation.showfileUploaderEncoding', False)
    st.set_option('deprecation.showfileUploaderEncoding', False)
    st.set_option('deprecation.showfileUploaderEncoding', False)
    st.set_option('deprecation.showfileUploaderEncoding', False)
    st.set_option('deprecation.showfileUploaderEncoding', False)
    st.set_option('deprecation.showfileUploaderEncoding', False)
    st.set_option('deprecation.showfileUploaderEncoding', False)
    st.set_option('deprecation.showfileUploaderEncoding', False)
    st.set_option('deprecation.showfileUploaderEncoding', False)
    st.set_option('deprecation.showfileUploaderEncoding', False)
    st.set_option('deprecation.showfileUploaderEncoding', False)
    st.set_option('deprecation.showfileUploaderEncoding', False)
    st.set_option('deprecation.showfileUploaderEncoding', False)
    st.set_option('deprecation.showfileUploaderEncoding', False)
    st.set_option('deprecation.showfileUploaderEncoding', False)
    st.set_option('deprecation.showfileUploaderEncoding', False)
    st.set_option('deprecation.showfileUploaderEncoding', False)
    st.set_option('deprecation.showfileUploaderEncoding', False)
    st.set_option('deprecation.showfileUploaderEncoding', False)
    st.set_option('deprecation.showfileUploaderEncoding', False)
    st.set_option('deprecation.showfileUploaderEncoding', False)
    st.set_option('deprecation.showfileUploaderEncoding', False)
    st.set_option('deprecation.showfileUploaderEncoding', False)
    st.set_option('deprecation.showfileUploaderEncoding', False)
    st.set_option('deprecation.showfileUploaderEncoding', False)
    st.set_option('deprecation.showfileUploaderEncoding', False)
    st.set_option('deprecation.showfileUploaderEncoding', False)
    st.set_option('deprecation.showfileUploaderEncoding', False)
    st.set_option('deprecation.showfileUploaderEncoding', False)
    st.set_option('deprecation.showfileUploaderEncoding', False)
    st.set_option('deprecation.showfileUploaderEncoding', False)