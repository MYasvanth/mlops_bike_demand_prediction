import pandas as pd
from zenml import step


@step
def data_ingestion(data_path: str) -> pd.DataFrame:
    """Load the bike demand dataset from CSV file."""
    df = pd.read_csv(data_path)
    return df
