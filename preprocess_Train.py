import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("emotion_dataset.csv")

# Convert labels to integers
label_map = {label: i for i, label in enumerate(df.iloc[:, -1].unique())}
df['label'] = df.iloc[:, -1].map(label_map)

X = df.iloc[:, :-2].values.astype(np.float32)  # All landmark columns
y = df['label'].values.astype(np.int32)        # Encoded labels

# Normalize (mean=0, std=1)
X = (X - X.mean(axis=0)) / X.std(axis=0)

# Train-test split
def train_test_split(X, y, test_ratio=0.2):
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    split = int(len(X) * (1 - test_ratio))
    return X[indices[:split]], X[indices[split:]], y[indices[:split]], y[indices[split:]]

X_train, X_test, y_train, y_test = train_test_split(X, y)

def softmax(z):
    exp = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp / np.sum(exp, axis=1, keepdims=True)

def one_hot(y, num_classes):
    return np.eye(num_classes)[y]

class SoftmaxClassifier:
    def __init__(self, input_dim, num_classes, lr=0.05):
        self.W = np.random.randn(input_dim, num_classes) * 0.01
        self.b = np.zeros((1, num_classes))
        self.lr = lr

    def train(self, X, y, epochs=4000):
        y_onehot = one_hot(y, self.b.shape[1])
        losses = []
        for epoch in range(epochs):
            logits = np.dot(X, self.W) + self.b
            probs = softmax(logits)
            loss = -np.mean(np.sum(y_onehot * np.log(probs + 1e-8), axis=1))
            losses.append(loss)

            # Backprop
            grad_logits = probs - y_onehot
            dW = np.dot(X.T, grad_logits) / X.shape[0]
            db = np.sum(grad_logits, axis=0, keepdims=True) / X.shape[0]

            self.W -= self.lr * dW
            self.b -= self.lr * db

            if epoch % 100 == 0:  # Print every 100 epochs
                print(f"Epoch {epoch}: Loss = {loss:.4f}")

        # Plot the loss over epochs
        plt.plot(losses)
        plt.title('Loss over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.show()

    def predict(self, X):
        logits = np.dot(X, self.W) + self.b
        return np.argmax(softmax(logits), axis=1)

# Train the model
model = SoftmaxClassifier(input_dim=X_train.shape[1], num_classes=len(label_map))
model.train(X_train, y_train, epochs=2000)

# Evaluate the model
preds = model.predict(X_test)
accuracy = np.mean(preds == y_test)
print(f"Test Accuracy: {accuracy*100:.2f}%")