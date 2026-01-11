import yfinance as yf
import pandas as pd

# 1. Hämta data
data = yf.download("EQQQ.L", start="2015-01-01")

# 2. Ta bort MultiIndex
data.columns = data.columns.get_level_values(0)

# 3. Behåll det vi behöver
data = data[["Close", "Volume"]]

# 4. Månadsmedel
monthly = data.resample("M").mean()

# 5. Nästa månad
monthly["Next_Close"] = monthly["Close"].shift(-1)

# 6. Label: upp eller ner
monthly["Up_Down"] = (monthly["Next_Close"] > monthly["Close"]).astype(int)

# 7. Ta bort sista raden
monthly = monthly.dropna()

print(monthly.head())
