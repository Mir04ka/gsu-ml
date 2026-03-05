import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler

boston = fetch_openml(name="boston", version=1, as_frame=False)

X = boston.data.astype(np.float32)

y = boston.target.astype(np.float32).reshape(-1, 1)

# Нормализация
scaler = StandardScaler()
X = scaler.fit_transform(X)

train_features = X[:450]
train_prices   = y[:450]

test_features  = X[450:500]
test_prices    = y[450:500]

n_features = train_features.shape[1]

# y = XW + b 
W = tf.Variable(tf.random.normal([n_features, 1]))
b = tf.Variable(tf.zeros([1]))

def predict(X):
    return tf.matmul(X, W) + b

# среднеквадратичная ошибка
def mse(y_pred, y_true):
    return tf.reduce_mean(tf.square(y_pred - y_true))

learning_rate = 0.01
optimizer = tf.optimizers.SGD(learning_rate)

epochs = 400
history = []
# 200 - 9.5; 2.5
# 400 - 7.6; 2.3
# 600 - 7.9; 2.3
# 800 - 8.0; 2.3

for epoch in range(epochs):
    with tf.GradientTape() as tape:
        y_pred = predict(train_features)
        loss = mse(y_pred, train_prices)

    gradients = tape.gradient(loss, [W, b])

    optimizer.apply_gradients(zip(gradients, [W, b]))

    history.append(loss.numpy())

    if epoch % 20 == 0:
        print("Epoch:", epoch, "Loss:", loss.numpy())

# тест
test_predictions = predict(test_features)

test_mse = mse(test_predictions, test_prices)
test_mae = tf.reduce_mean(tf.abs(test_predictions - test_prices))

print("\nTest MSE:", test_mse.numpy())
print("Test MAE:", test_mae.numpy())

# Ошибки
plt.figure()
plt.plot(history)
plt.title("Training Loss (MSE)")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.show()

# Реальные и предсказанные значения
plt.figure()
plt.scatter(test_prices, test_predictions)
plt.xlabel("Real prices")
plt.ylabel("Predicted prices")
plt.title("Real vs predicted prices")
plt.gca().set_aspect('equal')
plt.plot([5, 35], [5, 35], 'r--', label='x = y')
plt.legend()
plt.show()