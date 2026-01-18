import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

# Download data
data = yf.download("EQQQ.L", start="2015-01-01")
data.columns = data.columns.get_level_values(0)
data = data[["Close", "Volume"]]

# Monthly aggregation
monthly = data.resample("M").agg({
    "Close": "last",
    "Volume": "mean"
})

# Features
monthly["Return_1m"] = monthly["Close"].pct_change()
monthly["MA_3"] = monthly["Close"].rolling(3).mean()
monthly["MA_6"] = monthly["Close"].rolling(6).mean()
monthly["Vol_3"] = monthly["Volume"].rolling(3).mean()
monthly["Return_3m"] = monthly["Close"].pct_change(3)

# Target
monthly["Next_Close"] = monthly["Close"].shift(-1)
monthly["Up_Down"] = (monthly["Next_Close"] > monthly["Close"]).astype(int)

monthly = monthly.dropna()

# Train/Test split
features = ["Close", "Volume", "Return_1m", "MA_3", "MA_6", "Vol_3", "Return_3m"]
X = monthly[features]
y = monthly["Up_Down"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# Model with class weighting
model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print(classification_report(y_test, y_pred))

# Visualization
plot_dates = monthly.index[-len(y_test):] + pd.offsets.MonthEnd(1)

plt.figure(figsize=(14, 6))

# Background: correct/wrong prediction
for date, actual, pred in zip(plot_dates, y_test.values, y_pred):
    color = 'green' if actual == pred else 'red'
    plt.axvspan(date - pd.Timedelta(days=15), date + pd.Timedelta(days=15), color=color, alpha=0.1)

# Actual
plt.scatter(plot_dates, y_test.values, label="Actual", color="blue", marker="o", s=100)

# Predicted
plt.scatter(plot_dates, y_pred, label="Predicted", color="orange", marker="x", s=100, alpha=0.7)

plt.yticks([0, 1], ["Down", "Up"])
plt.xlabel("Predicted month")
plt.ylabel("Direction")
plt.title(f"EQQQ Next-Month Prediction | Accuracy: {accuracy:.2f}")
plt.xticks(rotation=45)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
