import pandas as pd

df = pd.read_csv("data/store_sales.csv", encoding="latin1")

print(df.head())

print(df.info())

print("\nMissing values in each column:")
print(df.isnull().sum())

total_sales = df["Sales"].sum()
total_profit = df["Profit"].sum()

print("\nTotal Sales:", total_sales)
print("Total Profit:", total_profit)

sales_by_category = df.groupby("Category")["Sales"].sum()
print("\nSales by Category:")
print(sales_by_category)
import matplotlib.pyplot as plt

profit_by_category = df.groupby("Category")["Profit"].sum()

print("\nProfit by Category:")
print(profit_by_category)

profit_by_category.plot(kind="bar", title="Profit by Category")
plt.ylabel("Profit")
plt.xlabel("Category")
plt.tight_layout()
plt.show()
