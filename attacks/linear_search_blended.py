import torch
from utils import pert_to_full, bound_advs, determine_success

def linear_search_blended_uniform_noise_attack(
    model,
    x,
    features_pertubated,
    epsilon=0.20,
    directions=1000,
    steps=1000,
    device="cpu"
):
    """
    Linear Search Blended Uniform Noise Attack adapted for your codebase.

    Starts from uniform random noise that is adversarial, then blends towards the original input
    in linear steps until it ceases to be adversarial. Returns the minimal adversarial found.

    Args:
        model: PyTorch model in eval mode.
        x: Original inputs (batch_size x num_features).
        features_pertubated: Number of features allowed to be perturbed.
        epsilon: Relative perturbation bound for bound_advs().
        directions: Number of random noise draws to find an initial adversarial.
        steps: Number of linear interpolation steps between original and random adversarial.
        device: Torch device.
    Returns:
        x_adv, success, probability_of_benign, query_count
    """
    model.eval()
    # x = x.to(device)
    bs = x.size(0)
    query_count = torch.zeros(bs, dtype=torch.int64, device=device)

    # Step 1: Find initial adversarial noise
    min_, max_ = 0.0, 1.0
    random_adv_part = None
    found_mask = torch.zeros(bs, dtype=torch.bool, device=device)

    for j in range(directions):
        rand_part = torch.rand((bs, features_pertubated), device=device) * (max_ - min_) + min_
        rand_full = pert_to_full(rand_part, x, features_pertubated)
        rand_full = bound_advs(rand_full, x, features_pertubated, epsilon=epsilon)

        succ, _ = determine_success(model, rand_full)
        query_count += (~succ).int()

        if random_adv_part is None:
            random_adv_part = rand_part.clone()
            found_mask = succ.clone()
        else:
            # keep existing where already successful, otherwise replace with new successful
            replace_mask = ~found_mask & succ
            random_adv_part[replace_mask] = rand_part[replace_mask]
            found_mask |= succ

        if found_mask.all():
            break

    if not found_mask.any():
        raise RuntimeError("Failed to find any initial adversarial via random noise.")

    # Fill in unsuccessful cases with any successful example
    if not found_mask.all():
        successful_parts = random_adv_part[found_mask]
        idxs = torch.randint(0, successful_parts.size(0), (bs,), device=device)
        random_adv_part[~found_mask] = successful_parts[idxs[~found_mask]]

    # Step 2: Linear search from original to random adversarial
    orig_part = x[:, :features_pertubated].clone()
    best_eps = torch.ones(bs, device=device)  # track minimal blending factor

    for alpha in torch.linspace(0, 1, steps + 1, device=device):
        blended_part = (1 - alpha) * orig_part + alpha * random_adv_part
        blended_full = pert_to_full(blended_part, x, features_pertubated)
        blended_full = bound_advs(blended_full, x, features_pertubated, epsilon=epsilon)

        succ, _ = determine_success(model, blended_full)
        query_count += (~succ).int()

        # update best eps where success
        best_eps = torch.where(succ & (alpha < best_eps), alpha, best_eps)

        if (best_eps < 1).all():
            break

    # Step 3: Construct final adversarial examples
    final_part = (1 - best_eps.view(-1, 1)) * orig_part + best_eps.view(-1, 1) * random_adv_part
    final_full = pert_to_full(final_part, x, features_pertubated)
    final_full = bound_advs(final_full, x, features_pertubated, epsilon=epsilon)

    success, prob_benign = determine_success(model, final_full)
    query_count += (~success).int()

    return final_full.detach(), success.detach(), prob_benign.detach(), query_count.detach()

