import torch
from attacks import *
from utils import generate_metrics, filter_attack_only
from config import TARGET_MODEL_CLASS, OUTPUT_DIM, SPLIT_RATIO, ATTACK_BATCH_SIZE

def attack(arg_attack, model, x, y, features_pertubated, epsilon=0.1, device=torch.device('cpu')):
    print("Starting "+arg_attack.__name__)
    with torch.no_grad():
        model = model.to(device)
        model.eval()
        # split the input into batches of size 1000
        batch_size = ATTACK_BATCH_SIZE
        num_batches = (x.size(0) + batch_size - 1) // batch_size
        for i in range(num_batches):
            print(f"Running attacks with batch {i + 1}/{num_batches}...")
            x_batch = x[i * batch_size:(i + 1) * batch_size]
            y_batch = y[i * batch_size:(i + 1) * batch_size]

            # send batch to device
            x_batch = x_batch.to(device).detach()
            y_batch = y_batch.to(device).detach()

            x_adv, success, probability_of_benign, query_count = arg_attack(model, x_batch, features_pertubated, epsilon=epsilon, device=device)

            x_adv = x_adv.detach().clone().to(torch.device('cpu'))
            success = success.detach().clone().to(torch.device('cpu'))
            probability_of_benign = probability_of_benign.detach().clone().to(torch.device('cpu'))
            query_count = query_count.detach().clone().to(torch.device('cpu'))

            if i == 0:
                all_x_adv = x_adv
                all_success = success
                all_probability_of_benign = probability_of_benign
                all_query_count = query_count
            else:
                all_x_adv = torch.cat((all_x_adv, x_adv), dim=0)
                all_success = torch.cat((all_success, success), dim=0)
                all_probability_of_benign = torch.cat((all_probability_of_benign, probability_of_benign), dim=0)
                all_query_count = torch.cat((all_query_count, query_count), dim=0)

    return all_x_adv, all_success, all_probability_of_benign, all_query_count

if __name__ == "__main__":
    # Load data & model
    # load the dataset
    X = torch.load('X.pth')
    y = torch.load('y.pth').long()

    split_idx = int(SPLIT_RATIO * len(X))
    X_test, y_test = X[split_idx:], y[split_idx:]

    X_test, y_test = filter_attack_only(X_test, y_test)
    model = TARGET_MODEL_CLASS(input_dim=X.shape[1], output_dim=OUTPUT_DIM)
    model.load_state_dict(torch.load(model.__class__.__name__ + "_model.pth"))
    model.eval()

    # Run attack
    x_adv, success, probability_of_benign, query_count = attack(linear_search_blended_uniform_noise_attack, model, X_test, y_test, 26, epsilon=0.20, device=torch.device('cuda'))
    metrics = generate_metrics(X_test, x_adv, success, probability_of_benign, query_count, 26)
    print(metrics)