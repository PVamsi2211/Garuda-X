REQUIRED_ACCOUNT_COLUMNS=["account_id","mrr"]

def validate_accounts(df):
    missing=[column for column in REQUIRED_ACCOUNT_COLUMNS if column not in df.columns]
    return {"valid":not missing,"missing_columns":missing}
