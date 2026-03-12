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

train_features = X[:400]
train_prices   = y[:400]

validation_features = X[400:450]
validation_prices = y[400:450]

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

epochs = 40000 # максимальное колчиество эпох
best_loss = float("inf") # минимальная ошибка за прошлые эпохи
failed_epochs = 0 # количество эпох, ошибка в которых выше, чем в лучшей
max_failed_epochs = 200
history = []
validation_history = []

for epoch in range(epochs):
    with tf.GradientTape() as tape:
        y_pred = predict(train_features)
        loss = mse(y_pred, train_prices)

    gradients = tape.gradient(loss, [W, b])

    optimizer.apply_gradients(zip(gradients, [W, b]))

    history.append(loss.numpy())

    validation_predictions = predict(validation_features)
    validation_loss = mse(validation_predictions, validation_prices)
    validation_history.append(validation_loss.numpy())

    if validation_loss < best_loss:
        best_loss = validation_loss
        failed_epochs = 0
    else:
        failed_epochs += 1

    if failed_epochs >= max_failed_epochs:
        print("Training stopping at epoch:", epoch)
        break

    if epoch % 20 == 0:
        print("Epoch:", epoch, "Loss:", loss.numpy(), "Validation loss: ", validation_loss.numpy())

# тест
test_predictions = predict(test_features)

test_mse = mse(test_predictions, test_prices)
test_mae = tf.reduce_mean(tf.abs(test_predictions - test_prices))

print("\nTest MSE:", test_mse.numpy())
print("Test MAE:", test_mae.numpy())

# Ошибки
plt.figure()
plt.plot(history, label="train")
plt.plot(validation_history, label="validation")
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