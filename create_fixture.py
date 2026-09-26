import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.normalization import normalize_source

# 6 rows covering:
# 1. plain US row
# 2. India row with a Devanagari/Gurmukhi name
# 3. row with a legal suffix
# 4. row with no address
# 5. row with a PIN code
# 6. row with a house number
# (They can overlap, but let's make 6 distinct or mostly distinct)

data = {
    'entity_id': ['S1-101', 'S1-102', 'S1-103', 'S1-104', 'S1-105', 'S1-106'],
    'country': ['US', 'India', 'India', 'US', 'US', 'US'],
    'business_name': [
        'Amazon Web Services', # plain US
        'अमेज़ॅन प्राइवेट लिमिटेड', # India devanagari
        'ABC Pvt. Ltd.', # legal suffix
        'Ghost Corp', # no address
        'Global Tech', # PIN code
        'Home Services' # house number
    ],
    'business_address': [
        'Seattle WA',
        'Mumbai',
        'New Delhi',
        '', # no address
        '10001 New York', # PIN code
        '123 Main St' # house number
    ]
}

df_raw = pd.DataFrame(data)
# Save raw
os.makedirs('samples', exist_ok=True)
df_raw.to_csv('samples/fixture_raw.tsv', sep='\t', index=False)

# Normalize
df_norm = normalize_source(df_raw)
df_norm.to_parquet('samples/fixture_normalized.parquet', index=False)
print("Fixture generation complete.")
