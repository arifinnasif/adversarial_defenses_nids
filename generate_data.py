import torch
from utils import load_data, reduce_benign_samples

# Load all data into X and y
X_, y_ = load_data('./datasets/Friday-02-03-2018_TrafficForML_CICFlowMeter.csv')
X_, y_ = reduce_benign_samples(X_, y_, 0.25)  # reduce benign samples to 25% of the original
X = X_
y = y_

# now concate the previous data with newly loaded data
X_, y_ = load_data('./datasets/Friday-23-02-2018_TrafficForML_CICFlowMeter.csv')
X_, y_ = reduce_benign_samples(X_, y_, 0.25)  # reduce benign samples to 25% of the original
X = torch.cat((X, X_), dim=0)
y = torch.cat((y, y_), dim=0)

X_, y_ = load_data('./datasets/Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv')
X_, y_ = reduce_benign_samples(X_, y_, 0.25)  # reduce benign samples to 25% of the original
X = torch.cat((X, X_), dim=0)
y = torch.cat((y, y_), dim=0)

X_, y_ = load_data('./datasets/Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv')
X_, y_ = reduce_benign_samples(X_, y_, 0.25)  # reduce benign samples to 25% of the original
X = torch.cat((X, X_), dim=0)
y = torch.cat((y, y_), dim=0)

X_, y_ = load_data('./datasets/Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv')
X_, y_ = reduce_benign_samples(X_, y_, 0.25)  # reduce benign samples to 25% of the original
X = torch.cat((X, X_), dim=0)
y = torch.cat((y, y_), dim=0)

X_, y_ = load_data('./datasets/Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv')
X_, y_ = reduce_benign_samples(X_, y_, 0.25)  # reduce benign samples to 25% of the original
X = torch.cat((X, X_), dim=0)
y = torch.cat((y, y_), dim=0)

# reduce benign from X, y
# X, y = reduce_benign_samples(X, y, 0.2)

# randomly shuffle the data
indices = torch.randperm(X.size(0))
X = X[indices]
y = y[indices].long()



# save the dataset
torch.save(X, 'X.pth')
torch.save(y, 'y.pth')

print(f"Saved dataset with {X.shape[0]} samples and {X.shape[1]} features.")
print(f"Saved labels with {y.shape[0]} samples.")
