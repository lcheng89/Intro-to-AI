import sys
import csv
import matplotlib.pyplot as plt
import numpy as np

# Q1: Load
def load_data(filename):
    years = []
    days = []
    with open(filename, 'r') as file:
        data = csv.reader(file)
        next(data)  # Skip header
        for row in data:
            years.append(int(row[0]))
            days.append(int(row[1]))
    return np.array(years), np.array(days)

# Q2: Visualize
def visualize(years, days):
    plt.figure()
    plt.plot(years, days)
    plt.xlabel("Year")
    plt.ylabel("Number of Frozen Days")
    plt.savefig("data_plot.jpg")
    plt.close()

# Q3: Normalization
def normalize(years):
    n = years.size
    m = np.min(years)
    M = np.max(years)
    normalized_years = (years - m) / (M - m)
    X_normalized = np.column_stack((normalized_years, np.ones(n)))  # nx2 array
    return X_normalized, m, M, n

# Q4: Closed-Form Linear Regression
def closed_form(X_normalized, Y):
    return np.linalg.inv(X_normalized.T @ X_normalized) @ X_normalized.T @ Y

# Q5: Gradient Descent
def gradient_descent(X_normalized, n, Y, learning_rate, iterations):
    w, b = 0, 0
    mse_history = []
    for i in range(iterations):
        if i % 10 == 0:
            print(np.array([w, b]))
        y_hat = w * X_normalized[:, 0] + b
        dw = (-2/n) * np.sum((Y - y_hat) * X_normalized[:, 0])
        db = (-2/n) * np.sum(Y - y_hat)
        w = w - learning_rate * dw
        b = b - learning_rate * db
        mse = np.mean((Y - y_hat) ** 2)
        mse_history.append(mse)
    return w, b, mse_history

def plot_loss(mse_history):
    plt.figure()
    plt.plot(range(len(mse_history)), mse_history)
    plt.xlabel("Iterations")
    plt.ylabel("MSE Loss")
    plt.savefig("loss_plot.jpg")
    plt.close()

# # Q6: Prediction
# def predict(w, b, year, m, M):
#     normalized_year = (year - m) / (M - m)
#     return w * normalized_year + b

if __name__ == "__main__":
    filename = sys.argv[1]
    learning_rate = float(sys.argv[2])
    iterations = int(sys.argv[3])

    # Load and normalize data
    years, days = load_data(filename)

    # Question 2: Plot the data
    visualize(years, days)

    # Question 3: Normalize and print the normalized matrix
    X_normalized, m, M, n = normalize(years)
    print("Q3:")
    print(X_normalized)

    # Question 4: Closed-form solution
    Y = days.reshape(-1, 1)
    theta_closed = closed_form(X_normalized, Y)
    print("Q4:")
    print(theta_closed.flatten())
    [weights, bias]=theta_closed

    # Question 5: Gradient descent
    print("Q5a:")
    w, b, mse_history = gradient_descent(X_normalized, n, Y, learning_rate, iterations)
    plot_loss(mse_history)
    print(f"Q5b: {learning_rate}")
    print(f"Q5c: {iterations}")

    # Question 6: Prediction for 2023-24
    # y_hat = predict(weights, bias, 2023, m, M)
    y_hat = theta_closed[0] * (2023 - m) / (M - m) + theta_closed[1]
    print(f"Q6: {y_hat}")

    # Question 7a: Print the sign of the weight
    symbol = ">" if weights > 0 else ("<" if weights < 0 else "=")
    print(f"Q7a: {symbol}")

    # Question 7b: Interpretation of the sign
    interpretation = "If w > 0, the number of frozen days is increasing over time. If w < 0, the number of frozen days is decreasing over time. If w = 0, there is no change in the number of frozen days over time."
    print(f"Q7b: {interpretation}")

    # Question 8a: Predict when Lake Mendota will no longer freeze
    if weights != 0:
        x_star = m + (-bias * (M - m)) / weights
        print(f"Q8a: {x_star}")
    else:
        print("Q8a: No meaningful prediction, since w = 0")

    # Question 8b: Discuss limitations
    limitations = "This prediction assumes a linear trend which may not hold in reality. Climate change could accelerate or decelerate the trend. The model doesn't account for natural variability or potential tipping points in the climate system. Additionally, the prediction extrapolates far beyond the range of our data, which increases uncertainty."
    print(f"Q8b: {limitations}")