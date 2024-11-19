
- 👋 Hi, I’m @komen619
- 👀 I’m interested in ...
- 🌱 I’m currently learning ...
- 💞️ I’m looking to collaborate on ...
- 📫 How to reach me ...
- 😄 Pronouns: ...
- ⚡ Fun fact: ...

<!---
komen619/komen619 is a ✨ special ✨ repository because its `README.md` (this file) appears on your GitHub profile.
You can click the Preview link to take a look at your changes.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pickle
# Load dataset
df = pd.read_csv('accident_data.csv')

# Handle missing data and convert categorical variables to numerical
df = pd.get_dummies(df, drop_first=True)

# Define dependent and independent variables
X = df.drop(columns=['Accident Severity'])
y = df['Accident Severity']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the model for future use
with open('accident_severity_model.pkl', 'wb') as f:
    pickle.dump(model, f)
# Make predictions and evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"MSE: {mse}, R²: {r2}")
# Example input for a new accident
new_data = {
    'Speed Limit': [50],
    'Traffic Volume': [2],  # Medium
    'Weather_Sunny': [1],   # Encoding for Sunny weather
    'Road Conditions_Wet': [1],  # Wet road condition
    'Location_Urban': [1]   # Urban location
}

new_data_df = pd.DataFrame(new_data)

# Predict accident severity
predicted_severity = model.predict(new_data_df)
print(f"Predicted Accident Severity: {predicted_severity[0]}")
# Load the saved model
with open('accident_severity_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# Use the model to make predictions
predicted_severity = loaded_model.predict(new_data_df)
print(f"Predicted Accident Severity: {predicted_severity[0]}")

--->
