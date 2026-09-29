from pathlib import Path
import pandas as pd

def load_csv(path):
    return pd.read_csv(Path(path))
