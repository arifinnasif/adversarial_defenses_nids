import torch
from utils import pert_to_full, bound_advs, determine_success

def pointwise_attack(
    model,
    x,
    features_pertubated,
    epsilon=0.20,
    l2_binary_search=True,
    init_attempts=50,
    device="cpu"
):
    """
    Pointwise attack adapted for your codebase.
    Starts from an adversarial example and tries to revert individual features to original values
    while keeping it adversarial.
    """
    model.eval()
    # x = x.to(device)
    bs = x.size(0)
    query_count = torch.zeros(bs, dtype=torch.int64, device=device)

    # ---- Step 1: Find starting adversarial examples ----
    x_adv = torch.zeros_like(x, device=device)
    found = False
    attempts = 0
    while attempts < init_attempts:
        attempts += 1
        rand_pert = torch.rand((bs, features_pertubated), device=device)
        cand_full = pert_to_full(rand_pert, x, features_pertubated)
        cand_full = bound_advs(cand_full, x, features_pertubated)
        succ, _ = determine_success(model, cand_full)
        query_count += (~succ).int()
        if succ.any():
            indices = torch.randint(0, cand_full[succ].size(0), (bs,), device=device)
            x_adv = cand_full[succ][indices % cand_full[succ].size(0)].clone()
            found = True
            break
    if not found:
        raise RuntimeError("Failed to find starting adversarial examples.")

    # ---- Step 2: Pointwise feature reverting ----
    orig = x.clone()
    adv_part = x_adv[:, :features_pertubated].clone()
    orig_part = orig[:, :features_pertubated].clone()

    for i in range(features_pertubated):
        # Try setting feature i back to original for all samples
        test_part = adv_part.clone()
        test_part[:, i] = orig_part[:, i]
        test_full = pert_to_full(test_part, orig, features_pertubated)
        test_full = bound_advs(test_full, orig, features_pertubated)
        succ, _ = determine_success(model, test_full)
        query_count += (~succ).int()
        # Keep revert if still adversarial
        adv_part[succ, i] = orig_part[succ, i]

    x_adv = pert_to_full(adv_part, orig, features_pertubated)

    # ---- Step 3: Optional L2 binary search refinement ----
    if l2_binary_search:
        for i in range(features_pertubated):
            adv_vals = adv_part[:, i].clone()
            orig_vals = orig_part[:, i].clone()
            for _ in range(10):
                mid_vals = (adv_vals + orig_vals) / 2
                test_part = adv_part.clone()
                test_part[:, i] = mid_vals
                test_full = pert_to_full(test_part, orig, features_pertubated)
                test_full = bound_advs(test_full, orig, features_pertubated)
                succ, _ = determine_success(model, test_full)
                query_count += (~succ).int()
                adv_vals = torch.where(succ, mid_vals, adv_vals)
                orig_vals = torch.where(succ, orig_vals, mid_vals)
            adv_part[:, i] = adv_vals
        x_adv = pert_to_full(adv_part, orig, features_pertubated)

    # ---- Step 4: Return results ----
    x_adv = bound_advs(x_adv, orig, features_pertubated)
    success, prob_benign = determine_success(model, x_adv)
    return x_adv.detach(), success.detach(), prob_benign.detach(), query_count.detach()
