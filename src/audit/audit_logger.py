from datetime import datetime,timezone

def create_audit_record(account_id,decision,status):
    return {"account_id":account_id,"decision":decision,"status":status,"timestamp":datetime.now(timezone.utc).isoformat()}
