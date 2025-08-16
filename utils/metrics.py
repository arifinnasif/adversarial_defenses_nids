def generate_metrics(x, x_adv, success, probability_of_benign, query_count, features_pertubated):
    print("Generating metrics...")
    success_ = success.float()
    asr = success_.mean()
    mape = 0
    for i in range(features_pertubated):
        mape += ((x_adv[:, i] - x[:, i]) / (x[:, i]+1e-10)).abs().mean()
    mape /= features_pertubated
    if success.any():
        benign_confidence = probability_of_benign[success].mean().item()
        query_rate = query_count[success].float().mean().item()
    else:
        benign_confidence = float('nan')
        query_rate = float('nan')
    return {
        "asr": asr.item(),
        "mape": mape.item(),
        "benign_confidence": benign_confidence,
        "query_rate": query_rate
    }

