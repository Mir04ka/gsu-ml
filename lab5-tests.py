import tensorflow as tf
import numpy as np
import csv
from sklearn.metrics import f1_score

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)

x_train = x_train.reshape(-1, 784).astype("float32") / 255.0
x_test = x_test.reshape(-1, 784).astype("float32") / 255.0

input_size = 784
output_size = 10

learning_rate = 0.01
batch_size = 64
epochs = 5

optimizer_class = tf.optimizers.SGD

sizes = [32, 48, 64, 96, 128]

results = []

def train_and_eval(h1_size, h2_size):
    W1 = tf.Variable(tf.random.normal([input_size, h1_size], stddev=0.1))
    b1 = tf.Variable(tf.zeros([h1_size]))

    W2 = tf.Variable(tf.random.normal([h1_size, h2_size], stddev=0.1))
    b2 = tf.Variable(tf.zeros([h2_size]))

    W3 = tf.Variable(tf.random.normal([h2_size, output_size], stddev=0.1))
    b3 = tf.Variable(tf.zeros([output_size]))

    optimizer = optimizer_class(learning_rate)

    def forward(x):
        h1 = tf.nn.relu(tf.matmul(x, W1) + b1)
        h2 = tf.nn.relu(tf.matmul(h1, W2) + b2)
        return tf.matmul(h2, W3) + b3

    def loss_fn(logits, labels):
        return tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits, labels=labels
            )
        )

    # Обучение
    for epoch in range(epochs):
        indices = np.random.permutation(len(x_train))
        x_shuffled = x_train[indices]
        y_shuffled = y_train[indices]

        num_batches = len(x_train) // batch_size

        for i in range(num_batches):
            x_batch = x_shuffled[i * batch_size:(i + 1) * batch_size]
            y_batch = y_shuffled[i * batch_size:(i + 1) * batch_size]

            with tf.GradientTape() as tape:
                logits = forward(x_batch)
                loss = loss_fn(logits, y_batch)

            grads = tape.gradient(loss, [W1, b1, W2, b2, W3, b3])
            optimizer.apply_gradients(zip(grads, [W1, b1, W2, b2, W3, b3]))

    # Оценка
    logits_test = forward(x_test)
    predictions = tf.argmax(logits_test, axis=1).numpy()

    return f1_score(y_test, predictions, average='macro')


for h1 in sizes:
    for h2 in sizes:
        scores = []

        for run in range(2):
            f1 = train_and_eval(h1, h2)
            scores.append(f1)

        avg_f1 = np.mean(scores)

        print(f"h1={h1}, h2={h2} → F1_avg={avg_f1:.4f}")

        results.append([h1, h2, avg_f1])


with open("results.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["hidden1", "hidden2", "f1_macro_avg"])
    writer.writerows(results)


print("\nRes")
for row in results:
    print(f"h1={row[0]:3}, h2={row[1]:3} → F1={row[2]:.4f}")