import pandas as pd
from sklearn.datasets import fetch_openml
import os

def download_german_credit_data():
    print("Downloading German Credit dataset from OpenML...")
    try:
        # Fetching the German Credit dataset (credit-g)
        data = fetch_openml(name='credit-g', version=1, as_frame=True)
        df = data.frame
        
        os.makedirs('data', exist_ok=True)
        output_path = 'data/german_credit_data.csv'
        df.to_csv(output_path, index=False)
        print(f"Dataset successfully downloaded and saved to: {output_path}")
        print(f"Dataset shape: {df.shape}")
    except Exception as e:
        print(f"Error downloading dataset: {e}")

if __name__ == "__main__":
    download_german_credit_data()
