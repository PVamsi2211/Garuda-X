from src.data.validator import validate_accounts

def test_validate_accounts_requires_core_columns():
    result=validate_accounts({"account_id":[]})
    assert result["valid"] is False
    assert "mrr" in result["missing_columns"]
