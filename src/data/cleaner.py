import pandas as pd

def clean_accounts(df):
    result=df.copy()
    if "mrr" in result.columns:
        result["mrr"]=pd.to_numeric(result["mrr"],errors="coerce")
    return result
