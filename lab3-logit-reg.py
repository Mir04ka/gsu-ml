import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# None:
# 0.945
# 0.922
# 0.965
# 0.954
# 0.936
# 0.951
# 0.960
# 0.957
# AVG: 0.94875

# 0
# 0.922
# 0.905
# 0.905

# 1
# 0.908
# 0.931
# 0.908

# 2
# 0.945
# 0.945
# 0.931
# 0.945
# 0.960
# 0.957
# 0.957
# 0.936
# AVG: 0.947

# 3
# 0.769
# 0.783
# 0.772

# 4
# 0.910
# 0.908
# 0.919

# 5
# 0.783
# 0.746
# 0.757

data = pd.read_csv('car_evaluation.csv', header=None)
# data = data.drop(columns=[2])

features = pd.get_dummies(data.iloc[:, :-1]).values

labels_raw = data.iloc[:, -1]

mapping = {
    'unacc': 0,
    'acc': 1,
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
        print(f"Epoch {epoch}")

predictions = calc_predictions(test_features).numpy()
predictions_classes = (predictions > 0.5).astype(int)


points_test = [[], [], [], []]
points_test_err = [[], []]
right_yes = 0
right_no = 0
total_yes = 0
total_no = 0

for i in range(len(test_classes)):
    true = test_classes[i][0]
    pred = predictions_classes[i][0]
    prob = predictions[i][0]

    err = abs(true - prob)

    points_test_err[0].append(i + 1)
    points_test_err[1].append(err)
    points_test[0].append(i + 1)
    points_test[1].append(true)
    points_test[2].append(prob)
    points_test[3].append(pred)
    
    if true == 1:
        total_yes += 1
        if pred == 1:
            right_yes += 1
    else:
        total_no += 1
        if pred == 0:
            right_no += 1

total = total_yes + total_no
right = right_yes + right_no

print(f"Total accuracy: {right} / {total} = {right / total:.3f}")

print(f"Yes accuracy: {right_yes} / {total_yes} = {right_yes / total_yes:.3f}")
print(f"No  accuracy: {right_no} / {total_no} = {right_no / total_no:.3f}")

print(f"Avg error: {(total - right) / total:.3f}")

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