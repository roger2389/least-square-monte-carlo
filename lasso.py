import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from scipy.stats import norm
from IPython.display import display, Latex

def stock_simu(S, T, r, sigma, N, M, seed):
    dt = T / N
    np.random.seed(seed)  
    paths = np.zeros((N + 1, M))
    paths[0] = S

    for t in range(1, N + 1):
        Z = np.random.standard_normal(int(M/2))
        Z = np.concatenate((Z, -Z))
        paths[t] = paths[t - 1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
    return paths

def call(ST, K):
    return np.maximum(ST - K, 0)

def put(ST, K):
    return np.maximum(K - ST, 0)

def lasso_lsmc(S, K, T, r, sigma, M, N, alpha, option='put', seed=123):
    payoff = {'call': call, 'put': put}[option]
    df = np.exp(-r * (T/N))
    stock_paths = stock_simu(S, T, r, sigma, N, M, seed)

    payoffs = payoff(stock_paths, K)

    exercise_values = np.zeros_like(payoffs)
    exercise_values[-1] = payoffs[-1]

    for t in range(N-1, 0, -1):
        in_the_money = payoffs[t] >= 0
        X = stock_paths[t][in_the_money].reshape(-1, 1)
        y = exercise_values[t+1][in_the_money] * df

        # Perform lasso regression
        lasso_model = Lasso(alpha=alpha)
        lasso_model.fit(X, y)

        # Calculate continuation values using the lasso model
        C = lasso_model.predict(X)
        exercise_values[t][in_the_money] = np.where(payoffs[t][in_the_money] > C,
                                                    payoffs[t][in_the_money],
                                                    exercise_values[t+1][in_the_money] * df
                                                    )
        exercise_values[t][~in_the_money] = payoffs[t+1][~in_the_money] * df

    return np.mean(exercise_values[1] * df), np.std(exercise_values[1] * df) / M**0.5

S = 36
K = 40
T = 1
r = 0.06
sigma = 0.2
N = 50
option = 'call'
M = [100, 500, 1000, 2000, 5000, 10000, 20000, 50000]
alpha = 0.1  # Lasso regression regularization parameter

mc_mean = []
mc_std = []
for n in M:
    m, s = lasso_lsmc(S, K, T, r, sigma, n, N, alpha)
    mc_mean.append(m)
    mc_std.append(s)

mc_mean = np.array(mc_mean)
mc_std = np.array(mc_std)

analysis_N = M
print('=================')
print('Pricing Value')
print('-----------------')
print(f'Monte Carlo mean: {mc_mean[-1]:.3f}')
print(f'Monte Carlo std: {mc_std[-1]:.3f}')

plt.figure(figsize=(10,5))
plt.plot(M, mc_mean, label='MC')
plt.plot(M, mc_mean+1.96*mc_std, label='mean+1.96 * std.', color='#E24A33', marker='o', linestyle='--')
plt.plot(M, mc_mean-1.96*mc_std, label='mean-1.96 * std.', color='#E24A33', marker='o', linestyle='--')
plt.grid()

plt.title(f'S={S}, K={K}, T={T}, r={r}, $\sigma$={sigma}, option={option}, LASSO', fontsize=14)
plt.xlabel('N', fontsize=14)
plt.ylabel('option value', fontsize=14)
plt.legend(fontsize=14)

plt.show()
