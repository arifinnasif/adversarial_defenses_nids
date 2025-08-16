import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
import torch
from config import TRAIN_MODEL_DEVICE, SPLIT_RATIO, TARGET_MODEL_CLASS, OUTPUT_DIM


# Load X.pth and y.pth
X = torch.load('X.pth')
y = torch.load('y.pth')

split_idx = int(len(X) * SPLIT_RATIO)
X_test, y_test = X[split_idx:], y[split_idx:]

with torch.no_grad():
    # Load the trained model
    target_model = TARGET_MODEL_CLASS(input_dim=X.shape[1], output_dim=OUTPUT_DIM)
    target_model.load_state_dict(torch.load(target_model.__class__.__name__ + '_model.pth'))
    target_model.to(TRAIN_MODEL_DEVICE)
    target_model.eval()

    y_pred = torch.softmax(target_model(X_test.to(TRAIN_MODEL_DEVICE)), dim=1).argmax(dim=1).cpu().numpy()

    # Evaluate the model
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
