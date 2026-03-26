import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('car_evaluation.csv', header=None)

features = pd.get_dummies(data.iloc[:, :-1]).values

labels_raw = data.iloc[:, -1]

mapping = {
    'unacc': 0,
    'acc': 0,
    'good': 1,
    'vgood': 1
}
labels = labels_raw.map(mapping).values.astype(np.float64)

classes = np.array([[y] for y in labels])

total_len = len(features)
indices = np.arange(total_len)
np.random.shuffle(indices)

features = features[indices]
classes = classes[indices]

scaler = StandardScaler()
X = scaler.fit_transform(features)

split = int(0.8 * total_len)
train_features = X[:split]
train_classes = classes[:split]
test_features = X[split:]
test_classes = classes[split:]

train_features = tf.constant(train_features, dtype=tf.float64)
train_classes = tf.constant(train_classes, dtype=tf.float64)
test_features = tf.constant(test_features, dtype=tf.float64)

num_features = train_features.shape[1]

A = tf.Variable(np.random.rand(num_features, 1), dtype=tf.float64)
b = tf.Variable(tf.zeros(1, dtype=tf.float64))

def calc_logits(x):
    return tf.add(b, tf.matmul(x, A))

def calc_predictions(x):
    return tf.nn.sigmoid(calc_logits(x))

def calc_error(x, targets):
    return tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=calc_logits(x),
            labels=targets
        )
    )

learning_rate = 0.01
epochs = 2000

for epoch in range(epochs):
    with tf.GradientTape() as tape:
        loss = calc_error(train_features, train_classes)

    gradients = tape.gradient(loss, [A, b])
    A.assign_sub(learning_rate * gradients[0])
    b.assign_sub(learning_rate * gradients[1])

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.numpy()}")

predictions = calc_predictions(test_features).numpy()
predictions_classes = (predictions > 0.5).astype(int)


points_test = [[], [], [], []]
points_test_err = [[], []]

for i in range(len(test_classes)):
    err = abs(test_classes[i] - predictions[i])
    points_test_err[0].append(i + 1)
    points_test_err[1].append(err[0])
    points_test[0].append(i + 1)
    points_test[1].append(test_classes[i][0])
    points_test[2].append(predictions[i][0])
    points_test[3].append(predictions_classes[i][0])

plt.subplot(3, 1, 1)
plt.plot(points_test[0], points_test[1], 'bo')
plt.plot(points_test[0], points_test[3], 'gx')
plt.title('testing result')
plt.ylabel('pred vs test')
plt.xlabel('n')

plt.subplot(3, 1, 2)
plt.plot(points_test[0], points_test[1], 'bo')
plt.plot(points_test[0], points_test[2], 'r+')
plt.title('testing result')
plt.ylabel('pred vs test')
plt.xlabel('n')

plt.subplot(3, 1, 3)
plt.plot(points_test_err[0], points_test_err[1], 'r-')
plt.xlabel('n')
plt.ylabel('diff')

plt.tight_layout()
plt.show()