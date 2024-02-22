import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Define tickers
tickers = ['AMZN', 'GOOGL', 'JPM', 'LLY', 'META', 'MSFT', 'NVDA', 'SMH', 'TGT', 'TSLA', 'V', 'AAPL', 'AVGO', 'AMD']
lookback = 300
end_day = datetime(2023, 2, 2)  # Specify your desired end date in YYYY, MM, DD format

# Fetch historical data for a broader period
start_date = (end_day - timedelta(days=lookback)).strftime('%Y-%m-%d')
data = yf.download(tickers, start=start_date, end=end_day.strftime('%Y-%m-%d'))['Adj Close']

# Split the data into two sections
split_date = (end_day - timedelta(days=(lookback)/6)).strftime('%Y-%m-%d')
data_first_half = data[:split_date]
data_second_half = data[split_date:]

# Calculate daily returns for the first half
daily_returns_first_half = data_first_half.pct_change().dropna()

# Optimization function
def calculate_sharpe_ratio(weights, returns):
    portfolio_return = np.dot(returns.mean() * 252, weights)
    portfolio_std = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
    return -portfolio_return / portfolio_std 

# Constraints and bounds
constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
bounds = tuple((0, 1) for _ in range(len(tickers)))

# Initial guess
initial_guess = np.array(len(tickers) * [1. / len(tickers)])

# Minimize the negative Sharpe Ratio for the first half
optimized_result = minimize(calculate_sharpe_ratio, initial_guess, args=(daily_returns_first_half,), method='SLSQP', bounds=bounds, constraints=constraints)
best_portfolio = optimized_result.x

# Apply the optimized weights to the second half
daily_returns_second_half = data_second_half.pct_change().dropna()
cumulative_returns_second_half = (1 + daily_returns_second_half.dot(best_portfolio)).cumprod()

# Round each weight to two decimal points
rounded_portfolio = [round(weight, 3) for weight in best_portfolio]

# Combine tickers with their corresponding weights
portfolio_with_tickers = list(zip(tickers, rounded_portfolio))

# Filter out stocks with 0% weight and sort in descending order by weight
filtered_and_sorted_portfolio = sorted([(ticker, weight) for ticker, weight in portfolio_with_tickers if weight > 0], key=lambda x: x[1], reverse=True)
print("Sharpe Ratio:", round(-optimized_result.fun, 3))

# Print the filtered and sorted portfolio
for ticker, weight in filtered_and_sorted_portfolio:
    print(f"{ticker}: {weight}")

# Plotting
plt.figure(figsize=(10, 6))

# Cumulative returns for the first half
cumulative_returns_first_half = (1 + daily_returns_first_half.dot(best_portfolio)).cumprod()
plt.plot(cumulative_returns_first_half, label='First Half', color='blue')

# Cumulative returns for the second half
plt.plot(cumulative_returns_second_half, label='Second Half', color='red')

# Add titles and labels
plt.title("Cumulative Returns of Best Portfolio")
plt.xlabel("Date")
plt.ylabel("Cumulative Returns")
plt.legend()

plt.show()


