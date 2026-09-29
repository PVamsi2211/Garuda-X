def predict_risk(model,features):
    return model.predict_proba(features)[:,1]
