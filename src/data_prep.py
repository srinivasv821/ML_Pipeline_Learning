import os
import pandas as pd
from sklearn.model_selection import train_test_split

# build path relative to repo root
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
data_path = os.path.join(BASE_DIR, "data", "raw", "spam.csv")

df = pd.read_csv(data_path, sep='\t', names=['label', 'message'])
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

train, test = train_test_split(df, test_size=0.2, random_state=42)

os.makedirs(os.path.join(BASE_DIR, "data", "processed"), exist_ok=True)
train.to_csv(os.path.join(BASE_DIR, "data", "processed", "train.csv"), index=False)
test.to_csv(os.path.join(BASE_DIR, "data", "processed", "test.csv"), index=False)
print("✅ Data split completed.")

