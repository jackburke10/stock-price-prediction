import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import yfinance as yf
import click

@click.command()
@click.option('--ticker', help='Ticket of the symbol you wish to run analysis on.')
@click.option('--start', help='Start date in yyyy-mm-dd format.')
@click.option('--end', help='Start date in yyyy-mm-dd format.')
def predict(ticker, start, end):
	# Fetch historical stock data
	stock_data = yf.download(ticker, start=start, end=end)

	# Feature engineering - calculate moving averages
	stock_data['MA_50'] = stock_data['Close'].rolling(window=50).mean()
	stock_data['MA_200'] = stock_data['Close'].rolling(window=200).mean()

	# Drop rows with missing values
	stock_data.dropna(inplace=True)

	# Define features and target variable
	X = stock_data[['Open', 'MA_50', 'MA_200']]
	y = stock_data['Close']

	# Split data into training and testing sets
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9, random_state=42)

	print(X_train)
	print(X_test)

	# Initialize and train the model
	model = LinearRegression()
	model.fit(X_train, y_train)

	# Make predictions
	predictions = model.predict(X_test)

	# Evaluate the model
	train_score = model.score(X_train, y_train)
	test_score = model.score(X_test, y_test)

	print(f'Training score: {train_score}')
	print(f'Testing score: {test_score}')

if __name__ == '__main__':
    predict()

