import torch
from config import BENIGN_ARG

def atleast_kd_torch(tensor, k):
    """
    Expand dims so that tensor has at least k dims by adding trailing singleton dims.
    """
    while tensor.ndim < k:
        tensor = tensor.unsqueeze(-1)
    return tensor

def pert_to_full(x_pert, x, features_pertubated):
    x_full = x.clone().detach()
    x_full[:, :features_pertubated] = x_pert
    return x_full

def bound_advs(x_adv, x, features_pertubated=26, epsilon=0.2):
    x_adv = x_adv.clone().detach()
    x_adv[:, :features_pertubated] = torch.min(torch.max(x_adv[:, :features_pertubated], x[:, :features_pertubated] * (1 - epsilon)), x[:, :features_pertubated] * (1 + epsilon))
    x_adv = x_adv.clamp(0, 1)
    return x_adv

def determine_success(model, x_adv):
    """
    Determine if the adversarial example is successful.
    Args:
        model: PyTorch model.
        x_adv: Adversarial example tensor.
    Returns:
        success: Boolean indicating if the attack was successful.
        probabilities of being benign
    """

    with torch.no_grad():
        logits = model(x_adv)
        probs = torch.softmax(logits, dim=1)
        success = (probs.argmax(dim=1) == BENIGN_ARG)
        
    success = torch.logical_and(success, x_adv[:,  0+2] >= x_adv[:,  0+3])
    success = torch.logical_and(success, x_adv[:,  0+0] <= x_adv[:,  0+2])
    success = torch.logical_and(success, x_adv[:,  0+0] >= x_adv[:,  0+3])

    success = torch.logical_and(success, x_adv[:,  4+2] >= x_adv[:,  4+3])
    success = torch.logical_and(success, x_adv[:,  4+0] <= x_adv[:,  4+2])
    success = torch.logical_and(success, x_adv[:,  4+0] >= x_adv[:,  4+3])

    success = torch.logical_and(success, x_adv[:,  8+2] >= x_adv[:,  8+3])
    success = torch.logical_and(success, x_adv[:,  8+0] <= x_adv[:,  8+2])
    success = torch.logical_and(success, x_adv[:,  8+0] >= x_adv[:,  8+3])

    success = torch.logical_and(success, x_adv[:, 12+2] >= x_adv[:, 12+3])
    success = torch.logical_and(success, x_adv[:, 12+0] <= x_adv[:, 12+2])
    success = torch.logical_and(success, x_adv[:, 12+0] >= x_adv[:, 12+3])

    success = torch.logical_and(success, x_adv[:, 16+2] >= x_adv[:, 16+3])
    success = torch.logical_and(success, x_adv[:, 16+0] <= x_adv[:, 16+2])
    success = torch.logical_and(success, x_adv[:, 16+0] >= x_adv[:, 16+3])

    return success, probs[:, 0]  # return success and the probability of the first class (Benign) for metrics calculation
