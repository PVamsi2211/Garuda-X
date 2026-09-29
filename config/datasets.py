from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    role: str
    directory: Path
    identifier: str
    files: dict[str, str]

    def path_for(self, file_key: str) -> Path:
        """Return the configured path for a named file in this dataset."""
        try:
            filename = self.files[file_key]
        except KeyError as error:
            known = ", ".join(sorted(self.files))
            raise KeyError(
                f"Unknown file key '{file_key}' for {self.name}; known keys: {known}"
            ) from error
        return self.directory / filename


DATASET_REGISTRY = {
    "synthetic_saas_churn": DatasetConfig(
        name="synthetic_saas_churn",
        role="primary churn-model dataset",
        directory=DATA_ROOT / "synthetic-saas-churn",
        identifier="user_id",
        files={
            "users": "users.parquet",
            "user_monthly": "user_monthly.parquet",
            "schema": "schema.json",
            "metadata": "meta.json",
        },
    ),
    "saas_revenue_risk": DatasetConfig(
        name="saas_revenue_risk",
        role="business and account intelligence",
        directory=DATA_ROOT / "saas-customer-churn-revenue-risk-analysis-main",
        identifier="account_id",
        files={
            "account_analysis": "account_analysis.csv",
            "subscription_revenue_analysis": "subscription_revenue_analysis.csv",
            "support_analysis": "support_analysis.csv",
            "usage_analysis": "usage_analysis.csv",
            "python_insights_summary": "python_insights_summary.csv",
        },
    ),
    "customer_churn_v2": DatasetConfig(
        name="customer_churn_v2",
        role="CRM interaction evidence reserved for later NLP retrieval",
        directory=DATA_ROOT / "customer-churn-v2",
        identifier="conversation_id",
        files={
            "train": "train.jsonl",
            "eval": "eval.jsonl",
            "full": "full.jsonl",
            "schema": "schema.json",
            "statistics": "stats.json",
        },
    ),
}


def get_dataset_path(dataset_name: str, file_key: str) -> Path:
    """Look up a dataset file path from the central registry."""
    try:
        dataset = DATASET_REGISTRY[dataset_name]
    except KeyError as error:
        known = ", ".join(sorted(DATASET_REGISTRY))
        raise KeyError(f"Unknown dataset '{dataset_name}'; known datasets: {known}") from error
    return dataset.path_for(file_key)
