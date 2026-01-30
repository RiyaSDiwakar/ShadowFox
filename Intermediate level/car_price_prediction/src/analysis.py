import pandas as pd

df = pd.read_csv("data/car_data.csv")

print(df.head())

df.drop("Car_Name", axis=1, inplace=True)

print(df.head())
# Convert text columns into numbers

df["Fuel_Type"] = df["Fuel_Type"].map({
    "Petrol": 0,
    "Diesel": 1,
    "CNG": 2
})

df["Seller_Type"] = df["Seller_Type"].map({
    "Individual": 0,
    "Dealer": 1
})

df["Transmission"] = df["Transmission"].map({
    "Manual": 0,
    "Automatic": 1
})

print(df.head())
# Convert text columns into numbers

df["Fuel_Type"] = df["Fuel_Type"].map({
    "Petrol": 0,
    "Diesel": 1,
    "CNG": 2
})

df["Seller_Type"] = df["Seller_Type"].map({
    "Individual": 0,
    "Dealer": 1
})

df["Transmission"] = df["Transmission"].map({
    "Manual": 0,
    "Automatic": 1
})

print(df.head())
# Separate input features and target

X = df.drop("Selling_Price", axis=1)
y = df["Selling_Price"]

print(X.head())
print(y.head())
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(X_train.shape, X_test.shape)
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(y_pred[:5])
from sklearn.metrics import mean_squared_error, r2_score

print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))
from sklearn.metrics import mean_squared_error, r2_score

print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))
# Predict price for a new car
sample_car = [[
    2018,   # Year
    5,      # Present price
    40000,  # Kilometers driven
    0,      # Owners
    1,      # Fuel type (Petrol)
    0,      # Seller type (Individual)
    1       # Transmission (Manual)
]]

predicted_price = model.predict(sample_car)
print("Predicted Car Price:", predicted_price)
