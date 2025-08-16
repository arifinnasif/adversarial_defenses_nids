import math
import torch
from utils import atleast_kd_torch, pert_to_full, bound_advs, determine_success

def hop_skip_jump_attack(
    model,
    x,
    features_pertubated,
    epsilon=0.20,
    steps=10,
    initial_gradient_eval_steps=100,
    max_gradient_eval_steps=1000,
    stepsize_search="geometric_progression",  # "geometric_progression" or "grid_search"
    gamma=1.0,
    constraint="l2",  # "l2" or "linf"
    init_attempts=50,
    device="cpu",
):
    """
    HopSkipJump-like decision-based attack implemented for your codebase.
    Relies on:
      - pert_to_full(pert_part, x, features_pertubated) -> full input tensor to model
      - determine_success(model, x_full) -> (success_bool_tensor, probability_of_benign_tensor)
      - BENIGN_ARG: label index for the benign/target class used in determine_success
    Args:
        model: PyTorch model (in eval mode).
        x: Input tensor (batch_size x C x H x W).
        features_pertubated: number of leading features (channels / flattened features) that may be changed.
            (This follows your NES implementation semantics — we perturb x[:, :features_pertubated]).
        steps: number of optimization iterations.
        initial_gradient_eval_steps: initial number of samples for gradient estimation.
        max_gradient_eval_steps: max number of samples for gradient estimation.
        stepsize_search: how to search for step size ("geometric_progression" or "grid_search").
        gamma: binary search threshold parameter.
        constraint: "l2" or "linf".
        init_attempts: attempts to find initial adversarial via random noise.
        device: torch device string.
    Returns:
        x_adv (batch_size x ...), success (bool tensor), probability_of_benign (tensor), query_count (int tensor)
    """

    model.eval()
    # x = x.to(device)
    bs = x.size(0)

    # Track queries per example
    query_count = torch.zeros(bs, dtype=torch.int64, device=device)

    
    # initialize x_adv_pert: try to find adversarial starting points by random perturbations around x's pert parts
    x_adv_pert = x[:, :features_pertubated].clone().detach()
    # found_adv = torch.zeros(bs, dtype=torch.bool, device=device)

    # If original is already adversarial, keep it (rare). We'll check original first.
    # full_orig = x
    # orig_success, orig_prob = determine_success(model, full_orig)
    # query_count.add_( (~orig_success).int() )
    # found_adv = orig_success.clone()

    # Try random init search to find a starting adversarial point per sample
    attempts = 0
    while attempts < init_attempts:
        attempts += 1
        # sample uniform noise in [0,1] for the perturbed portion then clamp
        rand_pert = torch.rand_like(x_adv_pert)
        cand_full = pert_to_full(rand_pert, x, features_pertubated)
        succ, prob = determine_success(model, cand_full)
        
        if succ.any():
            successful_rand_pert = rand_pert[succ]
            successful_cand_full = cand_full[succ]
            
            # now successful_cand_full can be of shape (any, x.size(1)). my goal is to make it (x.size(0), x.size(1)) by randomly copying
            indices = torch.randint(0, successful_cand_full.size(0), (x.size(0),), device=device)
            x_adv_pert = successful_rand_pert[indices].detach().clone()
            x_advs = successful_cand_full[indices].detach().clone()
            break  # we found at least one adversarial starting point

        del rand_pert
        del cand_full

    # if not found_adv.all():
    #     # Could not find starting adversarials for all samples
    #     failed = (~found_adv).sum().item()
    #     raise ValueError(f"init search failed to find adversarial for {failed} input(s)")

    # Compose starting x_adv
    # x_advs = pert_to_full(torch.rand_like(x_adv_pert), x, features_pertubated)
    # Project to boundary by binary search between original and start
    def binary_search_to_boundary(orig_full, adv_full):
        """
        For each sample, binary search along the line orig -> adv to find point on boundary.
        Returns perturbed full inputs at boundary.
        """
        nonlocal query_count
        # orig_full_old = orig_full.clone().detach()
        # orig_full = orig_full[found_adv]
        # adv_full = adv_full[found_adv]
        bs_ = orig_full.size(0)
        # We will perform per-sample binary search; vectorize using tensors.
        lows = torch.zeros(bs_, device=device)
        highs = torch.ones(bs_, device=device)  # t parameter in [0,1], adv = (1)*adv_full + (0)*orig_full
        # choose thresholds based on constraint and dimensionality
        d = int(torch.prod(torch.tensor(orig_full.shape[1:], device=device)))
        if constraint == "linf":
            thresholds = (linf_batch(orig_full, adv_full) * gamma / (d * d)).clamp(min=1e-8)
        else:
            thresholds = torch.ones(bs_, device=device) * (gamma / (d * math.sqrt(d)))
        old_mids = highs.clone()
        # loop until converged for all
        while True:
            mids = (lows + highs) / 2.0
            # create mids_full: (bs, ...) with broadcasting of mids
            mids_reshaped = mids.view(bs_, *([1] * (orig_full.ndim - 1)))
            mids_full = (1.0 - mids_reshaped) * orig_full + mids_reshaped * adv_full
            mids_full = bound_advs(mids_full, orig_full, features_pertubated)  # ensure within 20% of original
            succ, _ = determine_success(model, mids_full)
            query_count.add_( (~succ).int() )  # we called model; increment per-sample
            highs = torch.where(succ, mids, highs)
            lows = torch.where(succ, lows, mids)
            # check stop: highs - lows <= thresholds elementwise
            if torch.all(highs - lows <= thresholds):
                break
            # numerical precision guard
            if torch.all(old_mids == mids):
                break
            old_mids = mids.clone()
        mids_reshaped = highs.view(bs_, *([1] * (orig_full.ndim - 1)))
        res = (1.0 - mids_reshaped) * orig_full + mids_reshaped * adv_full
        # orig_full_old[found_adv] = res
        return res

    # utility: batch linf distance between originals and advs (used for threshold)
    def linf_batch(a, b):
        bs_local = a.size(0)
        return (a - b).abs().view(bs_local, -1).max(dim=1)[0]

    # utility: batch l2 distance
    def l2_batch(a, b):
        bs_local = a.size(0)
        return torch.norm((a - b).view(bs_local, -1), p=2, dim=1)
    # project initial advs to boundary
    x_advs = binary_search_to_boundary(x, x_advs)
    # recompute pert parts
    x_adv_pert = x_advs[:, :features_pertubated].clone().detach()

    # distance measure between original and current adv
    if constraint == "l2":
        distances = l2_batch(x, x_advs)
    else:
        distances = linf_batch(x, x_advs)

    # main optimization loop
    for step in range(steps):
        # select delta (step for finite-diff probing)
        if step == 0:
            delta = 0.1 * torch.ones_like(distances, device=device)
        else:
            d = int(torch.prod(torch.tensor(x.shape[1:], device=device)))
            if constraint == "linf":
                theta = gamma / (d * d)
                delta = d * theta * distances
            else:
                theta = gamma / (d * math.sqrt(d))
                delta = math.sqrt(d) * theta * distances
        # number of gradient estimation samples
        num_evals = int(min(initial_gradient_eval_steps * math.sqrt(max(1, step)), max_gradient_eval_steps))

        # approximate gradients (decision-only)
        # we follow the approach from Chen et al: sample gaussian directions, normalize, scale by delta
        # For batch: shape (num_evals, bs, ...pert_part shape)
        pert_shape = x_adv_pert.shape
        # We'll work in the perturbed-space (only the features_pertubated portion)
        rv = torch.randn((num_evals, *pert_shape), device=device)
        # normalize each noise vector (per-sample)
        rv_flat = rv.view(num_evals, bs, -1)
        norms = torch.norm(rv_flat, p=2, dim=2, keepdim=True)  # (num_evals, bs, 1)
        rv = rv_flat / (norms + 1e-12)
        # print("rv shape:", rv.shape)
        rv = rv.view(num_evals, *pert_shape)
        # print("rv shape after reshape:", rv.shape)

        # scaled_rv: scale each sample's directions by delta
        # print(pert_shape)
        # print("delta shape:", delta.shape)
        delta_expand = delta.view(1, bs, -1)
        scaled_rv = rv * delta_expand

        # create perturbed candidates in pert-space and get decisions
        # Prepare perturbed full inputs for each sample and noise step
        # We'll collect multipliers (+1 if candidate is adversarial, -1 otherwise)
        multipliers = torch.zeros_like(rv, device=device)
        for i in range(num_evals):
            # print(x_adv_pert.shape)
            # print(scaled_rv.shape)
            pert_part = x_adv_pert.unsqueeze(0)[0] + scaled_rv[i]  # shape (bs, ...)
            pert_part = pert_part.clamp(0, 1)
            # print(pert_part.shape)
            # print(x.shape)
            # print(features_pertubated)
            full_cand = pert_to_full(pert_part, x, features_pertubated)
            # print(i)
            full_cand = bound_advs(full_cand, x, features_pertubated)
            succ, _ = determine_success(model, full_cand)
            query_count += (~succ).int()  # increment queries
            # set +1 for succ, -1 for not succ, per sample — expand to pert shape
            mult = torch.where(succ.view(bs, *([1] * (pert_part.ndim - 1))),
                               torch.ones_like(pert_part, device=device),
                               -torch.ones_like(pert_part, device=device))
            multipliers[i] = mult

        # vals: center multipliers (subtract mean across num_evals unless all the same)
        mean_mult = multipliers.mean(dim=0, keepdim=True)  # (1, bs, ...)
        all_same_mask = (mean_mult.abs() == 1)
        vals = torch.where(all_same_mask, multipliers, multipliers - mean_mult)

        # estimate gradient (in pert space)
        grad_est = (vals * (scaled_rv / (delta_expand + 1e-12))).mean(dim=0)  # (bs, ...)
        # normalize gradient
        grad_flat = grad_est.view(bs, -1)
        grad_norms = torch.norm(grad_flat, p=2, dim=1, keepdim=True)
        grad_est = (grad_flat / (grad_norms + 1e-12)).view_as(grad_est)

        if constraint == "linf":
            update = grad_est.sign()
        else:
            update = grad_est

        # stepsize search and update
        if stepsize_search == "geometric_progression":
            epsilons = distances / math.sqrt(step + 1 + 1e-12)  # avoid zero div
            # epsilons is (bs,) need to expand to pert shape, but for comparisons we keep per-sample
            epsilons_expand = epsilons.view(bs, *([1] * (x_adv_pert.ndim - 1)))
            while True:
                x_proposals_part = x_adv_pert + atleast_kd_torch(epsilons_expand, update.ndim) * update
                x_proposals_part = x_proposals_part.clamp(0, 1)
                full_props = pert_to_full(x_proposals_part, x, features_pertubated)
                full_props = bound_advs(full_props, x, features_pertubated)
                succ, _ = determine_success(model, full_props)
                query_count += (~succ).int()
                # where success is False, reduce eps by half
                epsilons = torch.where(succ, epsilons, epsilons / 2.0)
                epsilons_expand = epsilons.view(bs, *([1] * (x_adv_pert.ndim - 1)))
                if succ.any():
                    break
            # Update
            x_adv_pert = (x_adv_pert + atleast_kd_torch(epsilons_expand, update.ndim) * update).clamp(0, 1)
            x_advs = pert_to_full(x_adv_pert, x, features_pertubated)
            # return to boundary via binary search
            x_advs = binary_search_to_boundary(x, x_advs)
            x_adv_pert = x_advs[:, :features_pertubated].clone().detach()
        elif stepsize_search == "grid_search":
            # grid search on logspace for each sample (simplified vectorized)
            # build grid multipliers (20 values)
            grid = torch.logspace(-4, 0, steps=20, device=device).view(20, 1)
            # each eps grid: distances * grid_i
            proposals = []
            for g in range(grid.size(0)):
                eps_g = distances.view(bs, *([1] * (x_adv_pert.ndim - 1))) * grid[g]
                prop_part = (x_adv_pert + atleast_kd_torch(eps_g, update.ndim) * update).clamp(0, 1)
                prop_full = pert_to_full(prop_part, x, features_pertubated)
                prop_full = bound_advs(prop_full, x, features_pertubated)
                succ, _ = determine_success(model, prop_full)
                query_count += (~succ).int()
                # project to boundary
                prop_full_proj = binary_search_to_boundary(x, prop_full)
                # only use new values where initial guess was adversarial (succ)
                use_mask = succ.view(bs, *([1] * (prop_part.ndim - 1)))
                final_part = torch.where(use_mask, prop_full_proj[:, :features_pertubated], x_adv_pert)
                proposals.append(final_part)
            # choose minimal distance among proposals
            proposals_stack = torch.stack(proposals, dim=0)  # (20, bs, ...)
            # compute distances for each proposal
            # expand originals
            orig_expand = x.unsqueeze(0).expand(proposals_stack.shape[0], -1, -1, -1, -1)
            prop_fulls = torch.cat([pert_to_full(p, x, features_pertubated).unsqueeze(0) for p in proposals_stack], dim=0)
            # distances: compute per grid per sample
            if constraint == "l2":
                dist_vals = torch.norm((prop_fulls - orig_expand).view(prop_fulls.shape[0], bs, -1), p=2, dim=2)
            else:
                dist_vals = (prop_fulls - orig_expand).abs().view(prop_fulls.shape[0], bs, -1).max(dim=2)[0]
            min_idx = torch.argmin(dist_vals, dim=0)  # (bs,)
            # pick corresponding proposals
            new_parts = proposals_stack[min_idx, torch.arange(bs)]
            x_adv_pert = new_parts
            x_advs = pert_to_full(x_adv_pert, x, features_pertubated)
        else:
            raise ValueError("Unknown stepsize_search")

        # recompute distances
        if constraint == "l2":
            distances = l2_batch(x, pert_to_full(x_adv_pert, x, features_pertubated))
        else:
            distances = linf_batch(x, pert_to_full(x_adv_pert, x, features_pertubated))

    # Final success/prob
    x_adv = pert_to_full(x_adv_pert, x, features_pertubated)
    # ensure x_adv is within 20% of the original features
    x_adv = bound_advs(x_adv, x, features_pertubated)

    final_succ, final_prob = determine_success(model, x_adv)
    query_count += (~final_succ).int()

    
    # Return in the same structure as nes_attack
    return x_adv.detach(), final_succ.detach(), final_prob.detach(), query_count.detach()

