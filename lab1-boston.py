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


model = tf.keras.Sequential([
    tf.keras.layers.Dense(1) # Выход - цена
])

# Компиляция модели
model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
    loss='mse',
    metrics=['mae']
)

# Обучение модели
history = model.fit(
    train_features,
    train_prices,
    epochs=20,
    verbose=0
)

# Оценка
test_loss, test_mae = model.evaluate(test_features, test_prices, verbose=0)

print("Test MSE:", test_loss) # Среднеквадратичная ошибка
print("Test MAE:", test_mae) # Средняя абсолютная ошибка

# Предсказания
predictions = model.predict(test_features)

# Ошибки
plt.figure()
plt.plot(history.history['loss'])
plt.title("Training Loss (MSE)")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.show()

# Реальные и предсказанные значения
plt.figure()
plt.scatter(test_prices, predictions)
plt.xlabel("Real prices")
plt.ylabel("Predicted prices")
plt.title("Real vs predicted prices")
plt.gca().set_aspect('equal')
plt.plot([5, 35], [5, 35], 'r--', label='x = y')
plt.legend()
plt.show()