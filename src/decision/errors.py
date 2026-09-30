class DecisionError(Exception):
    pass


class ValidationError(DecisionError, ValueError):
    pass


class InvalidDecisionError(DecisionError, LookupError):
    pass


class InvalidTransitionError(DecisionError):
    pass


class AlreadyFinalizedError(InvalidTransitionError):
    pass


class ApprovalRequiredError(DecisionError):
    pass


class ExecutionNotAuthorizedError(DecisionError):
    pass


class AlreadyAuthorizedError(ExecutionNotAuthorizedError):
    pass


class AuditIntegrityError(DecisionError):
    pass
