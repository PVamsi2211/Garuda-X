def classify_risk(probability,high_threshold=0.70,medium_threshold=0.40):
    if probability>=high_threshold:
        return "High"
    if probability>=medium_threshold:
        return "Medium"
    return "Low"
