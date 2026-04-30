import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import f1_score

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)

# Нормализация
x_train = x_train.reshape(-1, 784).astype("float32") / 255.0
x_test = x_test.reshape(-1, 784).astype("float32") / 255.0

input_size = 784
hidden_size_1 = 48
hidden_size_2 = 128
output_size = 10

# Слои
W1 = tf.Variable(tf.random.normal([input_size, hidden_size_1], stddev=0.1))
b1 = tf.Variable(tf.zeros([hidden_size_1]))

W2 = tf.Variable(tf.random.normal([hidden_size_1, hidden_size_2], stddev=0.1))
b2 = tf.Variable(tf.zeros([hidden_size_2]))

W3 = tf.Variable(tf.random.normal([hidden_size_2, output_size], stddev=0.1))
b3 = tf.Variable(tf.zeros([output_size]))

learning_rate = 0.01
batch_size = 64
epochs = 10

train_losses = []

optimizer = tf.optimizers.SGD(learning_rate)

def forward(x):
    h1 = tf.nn.relu(tf.matmul(x, W1) + b1)
    h2 = tf.nn.relu(tf.matmul(h1, W2) + b2)
    logits = tf.matmul(h2, W3) + b3
    return logits

def loss_fn(logits, labels):
    return tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=labels
        )
    )

for epoch in range(epochs):
    indices = np.random.permutation(len(x_train))
    x_train = x_train[indices]
    y_train = y_train[indices]

    epoch_loss = 0
    num_batches = len(x_train) // batch_size

    for i in range(num_batches):
        x_batch = x_train[i * batch_size:(i + 1) * batch_size]
        y_batch = y_train[i * batch_size:(i + 1) * batch_size]

        with tf.GradientTape() as tape:
            logits = forward(x_batch)
            loss = loss_fn(logits, y_batch)

        grads = tape.gradient(loss, [W1, b1, W2, b2, W3, b3])
        optimizer.apply_gradients(zip(grads, [W1, b1, W2, b2, W3, b3]))

        epoch_loss += loss.numpy()

    epoch_loss /= num_batches
    train_losses.append(epoch_loss)

    print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")

plt.plot(train_losses)
plt.title("Loss during training")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid()
# plt.show()

logits_test = forward(x_test)
predictions = tf.argmax(logits_test, axis=1).numpy()

f1_macro = f1_score(y_test, predictions, average='macro')
f1_weighted = f1_score(y_test, predictions, average='weighted')

print(f"F1 (macro): {f1_macro:.4f}")
print(f"F1 (weighted): {f1_weighted:.4f}")

cm = confusion_matrix(y_test, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
# plt.show()