import torch
from utils import pert_to_full, bound_advs, determine_success

def salt_and_pepper_noise_attack(
    model,
    x,
    features_pertubated,
    epsilon=0.20,
    steps=1000,
    across_channels=True,  # not really used for tabular, kept for compatibility
    device="cpu"
):
    """
    Salt-and-Pepper Noise Attack adapted for your codebase.
    Gradually increases salt-and-pepper noise probability until inputs are misclassified.
    Returns: x_adv, success, prob_benign, query_count
    """
    model.eval()
    # x = x.to(device)
    bs = x.size(0)
    query_count = torch.zeros(bs, dtype=torch.int64, device=device)

    # bounds
    min_, max_ = 0.0, 1.0
    r = max_ - min_

    # start from original
    result = x.clone()
    success, _ = determine_success(model, result)
    query_count += (~success).int()

    # best L2 norms
    best_norms = torch.where(success, torch.zeros(bs, device=device), torch.full((bs,), float("inf"), device=device))

    # min and max probabilities for each sample
    min_p = torch.zeros(bs, device=device)
    max_p = torch.ones(bs, device=device)

    step_size = max_p / steps
    p = step_size.clone()

    for step in range(steps):
        # generate uniform random noise mask for the perturbed features
        u = torch.rand((bs, features_pertubated), device=device)

        salt = (u >= 1 - p.view(-1, 1) / 2).float() * r
        pepper = -(u < p.view(-1, 1) / 2).float() * r
        perturbed_part = (x[:, :features_pertubated] + salt + pepper).clamp(min_, max_)

        x_candidate = pert_to_full(perturbed_part, x, features_pertubated)
        x_candidate = bound_advs(x_candidate, x, features_pertubated, epsilon=epsilon)

        # check success
        succ, _ = determine_success(model, x_candidate)
        query_count += (~succ).int()

        # compute L2 norms
        norms = torch.norm((x_candidate - x).view(bs, -1), p=2, dim=1)
        closer = norms < best_norms
        is_best = succ & closer

        # update bests
        result = torch.where(is_best.view(bs, 1), x_candidate, result)
        best_norms = torch.where(is_best, norms, best_norms)
        min_p = torch.where(is_best, 0.5 * p, min_p)
        max_p = torch.where(is_best, torch.minimum(p * 1.2, torch.tensor(1.0, device=device)), max_p)

        remaining = steps - step
        step_size = torch.where(is_best, (max_p - min_p) / remaining, step_size)
        reset = p == max_p
        p = torch.where(is_best | reset, min_p, p)
        p = torch.minimum(p + step_size, max_p)

    result = bound_advs(result, x, features_pertubated, epsilon=epsilon)
    final_success, final_prob = determine_success(model, result)
    return result.detach(), final_success.detach(), final_prob.detach(), query_count.detach()
