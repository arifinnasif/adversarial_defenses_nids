import torch
from utils import pert_to_full, bound_advs, determine_success

def boundary_attack(
    model,
    x,
    features_pertubated,
    epsilon=0.20,
    steps=500,
    spherical_step_init=1e-2,
    source_step_init=1e-2,
    source_step_convergence=1e-7,
    step_adaptation=1.5,
    update_stats_every_k=10,
    init_attempts=100,
    device="cpu",
):
    """
    Boundary attack implementation adapted for your codebase.
    Returns: x_adv (batch, D), success (bool tensor), prob_benign (tensor), query_count (int tensor)
    Uses pert_to_full(...) and determine_success(...) from your file.
    """

    model.eval()
    # x = x.to(device)
    bs = x.size(0)

    # track queries per sample
    query_count = torch.zeros(bs, dtype=torch.int64, device=device)

    # originals and shapes
    originals = x.clone().detach()
    # shape = originals.shape

    # # initialize starting adversarials by random search on pert-part
    pert_shape = (bs, features_pertubated)
    # # use same dtype/device
    # rand_pert = torch.rand(pert_shape, device=device)
    # cand_full = pert_to_full(rand_pert, originals, features_pertubated)
    # cand_full = bound_advs(cand_full, x, features_pertubated)
    # succ, _ = determine_success(model, cand_full)
    # query_count += (~succ).int()  # we queried model once for all

    # found = succ.clone()
    x_adv_full = torch.zeros_like(originals, device=device)
    # x_adv_full[found] = cand_full[found]

    attempts = 0
    while attempts < init_attempts:
        attempts += 1
        rand_pert = torch.rand(pert_shape, device=device)
        cand_full = pert_to_full(rand_pert, originals, features_pertubated)
        cand_full = bound_advs(cand_full, x, features_pertubated)
        succ, _ = determine_success(model, cand_full)
        
        if succ.any():
            # successful_rand_pert = rand_pert[succ]
            successful_cand_full = cand_full[succ]
            
            # now successful_cand_full can be of shape (any, x.size(1)). my goal is to make it (x.size(0), x.size(1)) by randomly copying
            indices = torch.randint(0, successful_cand_full.size(0), (x.size(0),), device=device)
            x_adv_full[indices] = successful_cand_full[indices].detach().clone()
            # x_adv_pert[indices] = successful_rand_pert[indices].detach().clone()
            # found[indices] = True
            break

    

    # binary search each successful start to move to boundary
    def binary_search_to_boundary(orig, adv):
        nonlocal query_count
        # mask: which samples to operate on
        # idx = torch.where(mask)[0]
        # if len(idx) == 0:
        #     return orig_full
        # orig = orig_full[idx]
        # adv = adv_full[idx]
        lows = torch.zeros(orig.size(0), device=device)
        highs = torch.ones(orig.size(0), device=device)
        d = orig.view(orig.size(0), -1).size(1)
        # thresholds per-sample
        thresholds = torch.ones(orig.size(0), device=device) * 1e-6
        old_mids = highs.clone()
        while True:
            mids = (lows + highs) / 2.0
            mids_reshaped = mids.view(orig.size(0), *([1] * (orig.ndim - 1)))
            mids_full = (1.0 - mids_reshaped) * orig + mids_reshaped * adv
            mids_full = bound_advs(mids_full, x, features_pertubated)
            succ, _ = determine_success(model, mids_full)
            # print(query_count)
            query_count += (~succ).int()
            highs = torch.where(succ, mids, highs)
            lows = torch.where(succ, lows, mids)
            if torch.all(highs - lows <= thresholds):
                break
            if torch.all(old_mids == mids):
                break
            old_mids = mids.clone()
        mids_reshaped = highs.view(orig.size(0), *([1] * (orig.ndim - 1)))
        res = (1.0 - mids_reshaped) * orig + mids_reshaped * adv
        # out = orig.clone()
        # out[idx] = res
        return res

    x_adv_full = binary_search_to_boundary(originals, x_adv_full)

    # prepare best_advs (pert space and full)
    best_advs = x_adv_full.clone().detach()
    unnormalized_source = originals - best_advs
    source_norms = torch.norm(unnormalized_source.view(bs, -1), p=2, dim=1)
    source_directions = unnormalized_source / (source_norms.view(bs, *([1] * (originals.ndim - 1))) + 1e-12)

    spherical_steps = torch.ones(bs, device=device) * spherical_step_init
    source_steps = torch.ones(bs, device=device) * source_step_init

    # queues for stats (simple circular buffers implemented with lists of tensors)
    stats_sph = torch.zeros((100, bs), dtype=torch.bool, device=device)
    stats_sph_i = 0
    stats_step = torch.zeros((30, bs), dtype=torch.bool, device=device)
    stats_step_i = 0
    stats_sph_filled = False
    stats_step_filled = False

    bounds_min = 0.0
    bounds_max = 1.0

    for step_idx in range(1, steps + 1):
        # check convergence
        converged = source_steps < source_step_convergence
        if converged.all():
            break

        # recompute directions & norms
        unnormalized_source = originals - best_advs
        source_norms = torch.norm(unnormalized_source.view(bs, -1), p=2, dim=1)
        source_directions = unnormalized_source / (source_norms.view(bs, *([1] * (originals.ndim - 1))) + 1e-12)

        # draw proposals (vectorized)
        D = int(originals.view(bs, -1).size(1))
        # gaussian noise in flattened space for each sample
        eta = torch.randn((bs, D), device=device)
        # make orthogonal: subtract projection onto source_directions (flattened)
        sd_flat = source_directions.view(bs, -1)
        proj = (eta * sd_flat).sum(dim=1, keepdim=True) * sd_flat
        eta = eta - proj
        norms = torch.norm(eta, p=2, dim=1, keepdim=True)
        eta = eta * ((spherical_steps * source_norms).view(bs, 1) / (norms + 1e-12))
        # directions = eta - unnormalized_source_flat
        eta = eta.view_as(best_advs)  # (bs, ...)
        spherical_candidates = originals + (eta - unnormalized_source) / torch.sqrt(spherical_steps.view(bs, *([1] * (originals.ndim - 1))).square() + 1.0)
        # clip spherical candidates
        spherical_candidates = spherical_candidates.clamp(bounds_min, bounds_max)

        # step towards original (source step)
        new_source_dirs = originals - spherical_candidates
        new_source_norms = torch.norm(new_source_dirs.view(bs, -1), p=2, dim=1)
        lengths = source_steps * source_norms
        lengths = lengths + new_source_norms - source_norms
        lengths = torch.maximum(lengths, torch.zeros_like(lengths))
        lengths = lengths / (new_source_norms + 1e-12)
        lengths = lengths.view(bs, *([1] * (originals.ndim - 1)))
        candidates = spherical_candidates + lengths * new_source_dirs
        candidates = candidates.clamp(bounds_min, bounds_max)

        # evaluate candidates (decision)
        succ_cand, _ = determine_success(model, candidates)
        query_count += (~succ_cand).int()
        succ_sph, _ = determine_success(model, spherical_candidates)
        query_count += (~succ_sph).int()

        # update stats buffers (every update_stats_every_k)
        check_stats = (step_idx % update_stats_every_k == 0)
        if check_stats:
            stats_sph[stats_sph_i % stats_sph.shape[0]] = succ_sph
            stats_sph_i += 1
            if stats_sph_i >= stats_sph.shape[0]:
                stats_sph_filled = True
            stats_step[stats_step_i % stats_step.shape[0]] = succ_cand
            stats_step_i += 1
            if stats_step_i >= stats_step.shape[0]:
                stats_step_filled = True

        # accept candidates that are adversarial AND closer
        distances = torch.norm((originals - candidates).view(bs, -1), p=2, dim=1)
        closer = distances < source_norms
        is_best = succ_cand & closer
        # update best_advs where applicable and not yet converged
        update_mask = (~converged) & is_best
        if update_mask.any():
            best_advs[update_mask] = candidates[update_mask]

        # adapt steps if buffers full
        if stats_sph_filled:
            probs = stats_sph.float().mean(dim=0)
            cond_inc = probs > 0.5
            cond_dec = probs < 0.2
            spherical_steps = torch.where(cond_inc, spherical_steps * step_adaptation, spherical_steps)
            spherical_steps = torch.where(cond_dec, spherical_steps / step_adaptation, spherical_steps)
            source_steps = torch.where(cond_inc, source_steps * step_adaptation, source_steps)
            source_steps = torch.where(cond_dec, source_steps / step_adaptation, source_steps)
            # clear those slots by setting them to False where cond applied
            stats_sph[:, cond_inc | cond_dec] = False
            stats_sph_filled = False

        if stats_step_filled:
            probs2 = stats_step.float().mean(dim=0)
            cond_inc2 = probs2 > 0.25
            cond_dec2 = probs2 < 0.1
            source_steps = torch.where(cond_inc2, source_steps * step_adaptation, source_steps)
            source_steps = torch.where(cond_dec2, source_steps / step_adaptation, source_steps)
            stats_step[:, cond_inc2 | cond_dec2] = False
            stats_step_filled = False

    # final projection within epsilon of original (here epsilon interpreted as relative 20% by default)
    final = bound_advs(best_advs, originals, features_pertubated)

    final_succ, final_prob = determine_success(model, final)
    query_count += (~final_succ).int()

    return final.detach(), final_succ.detach(), final_prob.detach(), query_count.detach()
