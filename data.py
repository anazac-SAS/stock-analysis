import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

# 1. Download data
data = yf.download("EQQQ.L", start="2015-01-01")

# 2. Remove MultiIndex
data.columns = data.columns.get_level_values(0)

# 3. Keep only Close and Volume
data = data[["Close", "Volume"]]

# 4. Monthly averages
monthly = data.resample("M").mean()

# 5. Next month
monthly["Next_Close"] = monthly["Close"].shift(-1)

# 6. Up/Down label
monthly["Up_Down"] = (monthly["Next_Close"] > monthly["Close"]).astype(int)

# 7. Drop last row
monthly = monthly.dropna()

# -----------------------------
# Train/Test split
X = monthly[["Close", "Volume"]]
y = monthly["Up_Down"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Classification report
report = classification_report(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

# -----------------------------
# Visualization with classification report

plt.figure(figsize=(14,6))
plt.plot(y_test.values, label="Actual", marker='o')
plt.plot(y_pred, label="Predicted", marker='x', alpha=0.7)

# Add x-axis labels: years and months
dates = monthly.index[-len(y_test):]
plt.xticks(ticks=range(len(dates)), labels=[d.strftime("%Y-%m") for d in dates], rotation=45)

plt.title(f"EQQQ Up/Down Prediction\nAccuracy: {accuracy:.2f}")
plt.xlabel("Month")
plt.ylabel("Up=1 / Down=0")
plt.legend()

# Add classification report as text inside the plot
plt.text(0, -0.5, "Classification Report:\n" + report, fontsize=9, verticalalignment='top', fontfamily='monospace')

plt.tight_layout()
plt.show()
