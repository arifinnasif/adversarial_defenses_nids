import torch
import torch.nn as nn
from config import TRAIN_MODEL_DEVICE, OUTPUT_DIM, TARGET_MODEL_CLASS, SPLIT_RATIO



def train_model(model, X, y, epochs=10, batch_size=32, learning_rate=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # randomly shuffle the data
    indices = torch.randperm(X.size(0))
    X = X[indices]
    y = y[indices]

    # split the data into train and validation sets
    split_idx = int(0.8 * len(X))
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_val, y_val = X[split_idx:], y[split_idx:]

    min_val_loss = float('inf')


    

    
    for epoch in range(epochs):
        model.train()
        # shuffle the inputs and labels
        indices = torch.randperm(X_train.size(0))
        X_ = X_train[indices]
        y_ = y_train[indices]
        avg_epoch_loss = 0.0
        for i in range(0, len(X_), batch_size):
            # print(f'Epoch [{epoch + 1}/{epochs}], Batch [{i // batch_size + 1}/{len(X_) // batch_size}]')
            inputs = X_[i:i + batch_size]
            labels = y_[i:i + batch_size]


            # Ensure inputs and labels are on the same device
            inputs = inputs.to(TRAIN_MODEL_DEVICE)
            labels = labels.to(TRAIN_MODEL_DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            # print(outputs)
            # print(labels)
            # print(type(outputs[0]))
            # print(type(labels[0]))
            loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()
            avg_epoch_loss += loss.item()

            del(inputs)
            del(labels)
            del(outputs)
            del(loss)

        num_batches = (len(X_) + batch_size - 1) // batch_size
        # inside loop accumulate avg_epoch_loss
        avg_epoch_loss /= max(1, num_batches)
        print(f'Epoch [{epoch + 1}/{epochs}], Average Loss: {avg_epoch_loss:.4f}')
        # validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val.to(TRAIN_MODEL_DEVICE))
            val_loss = criterion(val_outputs, y_val.long().to(TRAIN_MODEL_DEVICE))
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                best_model = model.state_dict()
                # save the model
                torch.save(model.state_dict(), model.__class__.__name__ + '_model.pth')
            print(f'Validation Loss: {val_loss.item():.4f}')
        # print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

    model.load_state_dict(best_model)
    model.eval()
    print("Training complete.")
    return model

if __name__ == "__main__":
    X = torch.load('X.pth')
    y = torch.load('y.pth').long()

    # split the data into train and test sets
    split_idx = int(SPLIT_RATIO * len(X))
    X_train, y_train = X[:split_idx], y[:split_idx]
    # X_test, y_test = X[split_idx:], y[split_idx:]

    
    # # X_test, y_test = load_data('./Thursday-01-03-2018_TrafficForML_CICFlowMeter3.csv')
    # X_test, y_test = filter_attack_only(X_test, y_test)
    # X, y = load_data('./Friday-02-03-2018_TrafficForML_CICFlowMeter.csv')
    # X_test, y_test = load_data('./Thursday-01-03-2018_TrafficForML_CICFlowMeter3.csv')
    target_model = TARGET_MODEL_CLASS(input_dim=X.shape[1], output_dim=OUTPUT_DIM)

    target_model.to(TRAIN_MODEL_DEVICE)
    target_model = train_model(target_model, X_train, y_train, epochs=100, batch_size=4096*2, learning_rate=0.001)
    