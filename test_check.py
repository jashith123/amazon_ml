import pandas as pd

df = pd.read_parquet('samples/fixture_normalized.parquet')
for col in df.columns:
    print(f"--- {col} ---")
    for val in df[col]:
        if isinstance(val, str):
            print(val.encode('ascii', 'ignore').decode('ascii'))
        else:
            print(val)
