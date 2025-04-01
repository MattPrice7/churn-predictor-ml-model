import pandas as pd
import sqlite3

# Load the CSV
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Clean up the column names
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('-', '_')

# Create SQLite connection and load the data
conn = sqlite3.connect('telco_churn.db')
df.to_sql('customers', conn, if_exists='replace', index=False)

# Optional: preview a few rows
query = """
SELECT 
    contract,
    SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) AS churned_customers,
    1.0 * SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*) AS churn_rate
FROM customers
GROUP BY contract;
"""

df = pd.read_sql(query, conn)
print(df)

conn.close()