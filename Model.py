# Load model
data = np.load("softmax_model3.npz")
model = SoftmaxClassifier(input_dim=data['W'].shape[0], num_classes=data['W'].shape[1])
model.W = data['W']
model.b = data['b']

# Predict on new data
preds = model.predict(X_test)
print(f"Reloaded Model Accuracy: {np.mean(preds == y_test) * 100:.2f}%")
