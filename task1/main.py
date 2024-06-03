import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

true_theta = 2
k = 3
n_samples = 1000
sample_sizes = [10, 50, 100, 500, 1000]
threshold = 0.1

mean_biases = []
variances = []
mse = []
accuracy_counts = []


def generate_samples(theta, k, n_samples, sample_size):
    samples = np.random.gamma(shape=k, scale=theta, size=(n_samples, sample_size))
    return samples


def method_of_moments(sample):
    return np.mean(sample) / k


for sample_size in sample_sizes:
    biases = []
    squared_errors = []
    accuracy_count = 0

    samples = generate_samples(true_theta, k, n_samples, sample_size)
    for sample in samples:
        estimation = method_of_moments(sample)
        bias = estimation - true_theta
        squared_error = (estimation - true_theta) ** 2
        biases.append(bias)
        squared_errors.append(squared_error)
        if np.abs(bias) < threshold:
            accuracy_count += 1

    mean_bias = np.mean(biases)
    variance = np.var(biases)
    mse_value = np.mean(squared_errors)

    mean_biases.append(mean_bias)
    variances.append(variance)
    mse.append(mse_value)
    accuracy_counts.append(accuracy_count)

results_df = pd.DataFrame({
    'Размер выборки': sample_sizes,
    'Среднее отклонение': mean_biases,
    'Дисперсия': variances,
    'MSE': mse,
    'Количество точностей': accuracy_counts
})


def highlight_text(val):
    if isinstance(val, float):
        return f"\033[95m{val:.4f}\033[0m"
    else:
        return f"\033[95m{val}\033[0m"


print("Results:")
print(results_df.to_string(index=False, justify='left',
                           formatters={'Размер выборки': lambda x: f"{x:<15}", 'Среднее отклонение': highlight_text,
                                       'Дисперсия': highlight_text, 'MSE': highlight_text,
                                       'Количество точностей': highlight_text}, col_space=20))

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(sample_sizes, mse, marker='o')
plt.title('Среднеквадратическая ошибка')
plt.xlabel('Объем выборки')
plt.ylabel('MSE')

plt.subplot(1, 2, 2)
plt.plot(sample_sizes, accuracy_counts, marker='o')
plt.title('Количество точных оценок')
plt.xlabel('Объем выборки')
plt.ylabel('Количество')

plt.tight_layout()
plt.show()
