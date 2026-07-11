# Bank Customer Churn Analysis

# 1) Load Data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("Bank_Churn.csv")
dictionary = pd.read_csv("Bank_Churn_Data_Dictionary.csv")

# 2) Data Inspection

print(df.head())
print(df.shape)
print(df.columns)

print(df.info())
print(df.describe())
print(df.isnull().sum())

print(df["Exited"].value_counts())
print(df["Exited"].value_counts(normalize=True))

# 3) Data Cleaning

# Check for duplicates
print(df.duplicated().sum())

# 4) Exploratory Data Analysis (EDA)

print(df.groupby("Geography")["Exited"].agg(["count", "mean"]))
print(df.groupby("Gender")["Exited"].agg(["count", "mean"]))
print(df.groupby("NumOfProducts")["Exited"].agg(["count", "mean"]))
print(df.groupby("IsActiveMember")["Exited"].agg(["count", "mean"]))

# Key Findings:
# 1) Germany has approximately twice the churn rate of France and Spain
# 2) Female customers exhibit a higher churn rate than male customers
# 3) Inactive customers churn nearly twice as often as active customers
# 4) Customers with two products have the lowest churn rate
# 5) Customers with three or four products have very high churn rates,
# although these groups contain relatively few customers

# Summary Tables:

geo_summary = df.groupby("Geography")["Exited"].agg(["count","mean"])
gender_summary = df.groupby("Gender")["Exited"].agg(["count","mean"])
product_summary = df.groupby("NumOfProducts")["Exited"].agg(["count","mean"])
active_summary = df.groupby("IsActiveMember")["Exited"].agg(["count","mean"])

print(geo_summary)

# Business Questions

# Question #1: Age vs Churn
df["AgeGroup"] = pd.cut(
    df["Age"],
    bins=[18,30,40,50,60,100],
    labels=["18-30","31-40","41-50","51-60","60+"]
)

print(df.groupby("AgeGroup")["Exited"].agg(["count","mean"]))

# Findings: 
# France Churn rate: 16.2%
# Germany Churn rate: 32.4%
# Spain Churn rate: 16.7%

# Question #2: Credit Score
print(df.groupby(pd.cut(df["CreditScore"], bins=5))["Exited"].mean())

# Findings: 

# Lowest credit scores → around 32% churn
# Middle credit scores → around 20%
# Highest credit scores → around 20%

# Age Groups (suitable for Tableau): 
# 18-30: 7.5%
# 31-40: 12.1% 
# 41-50: 34%
# 51-60: 56.2%
# 60+: 24.8%

# Question #3: Balance
print(df.groupby(pd.cut(df["Balance"], bins=5))["Exited"].mean())

# Findings: 
# The highest balance bucket has roughly 59% churn

# Question #4: Estimated Salary
print(df.groupby(pd.cut(df["EstimatedSalary"], bins=5))["Exited"].mean())

# Findings: 
# The estimated salary does not appear to be a strong predictor of customer churn in this dataset

# Question #5: Tenure
print(df.groupby("Tenure")["Exited"].mean())

# Findings: 
# Customer tenure alone does not show a strong relationship with churn

# Question #6: Has Credit Card
print(df.groupby("HasCrCard")["Exited"].agg(["count","mean"]))

# Findings: 
# Possessing a credit card has little observable relationship with customer churn

# Question #7; Interaction
print(df.groupby(["Geography","Gender"])["Exited"]
      .agg(["count","mean"])
)

# Findings (approximate percentages): 

# Germany: 
# Female = 37.6%
# Male = 27.8%

# France: 
# Female = 20.3%
# Male = 12.7%

# Spain: 
# Female: 21.2%
# Male: 13.1%

# For example, German females have the highest churn

# 5) Feature Engineering

# 1) Age Group: 

# Feature #1: Age Groups

df["AgeGroup"] = pd.cut(
    df["Age"],
    bins=[18, 30, 40, 50, 60, 100],
    labels=["18–30", "31–40", "41–50", "51–60", "60+"],
    include_lowest=True
)

# Feature #2) Balance per Product: 

df["BalanceGroup"] = pd.cut(
    df["Balance"],
    bins=5
)
# Feature #3) High Credit Score: 

df["CreditCategory"] = pd.cut(
    df["CreditScore"],
    bins=[300,580,670,740,850],
    labels=[
        "Poor",
        "Fair",
        "Good",
        "Excellent"
    ]
)

# Feature #4) Activity Status:

df["ActivityStatus"] = df["IsActiveMember"].map({
    0:"Inactive",
    1:"Active"
})

# Feature #5) Product Category: 

df["ProductGroup"] = np.where(
    df["NumOfProducts"] >= 3,
    "Three+",
    df["NumOfProducts"].astype(str)
)

# 6) Tableau Export

print(df["ProductGroup"].value_counts(dropna=False))
print(df.tail())

df.to_csv(
    "Bank_Churn_Cleaned.csv",
    index=False
)