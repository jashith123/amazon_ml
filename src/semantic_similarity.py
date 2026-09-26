import argparse
import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    parser = argparse.ArgumentParser(description="Compute semantic similarity between name_norm pairs")
    parser.add_argument('--normalized', nargs='+', required=True, help='Paths to normalized records parquet(s)')
    parser.add_argument('--candidates', required=True, help='Path to candidate pairs parquet')
    parser.add_argument('--output', required=True, help='Path to output parquet')
    parser.add_argument('--batch-size', type=int, default=1024, help='Batch size for embeddings')
    args = parser.parse_args()

    # Read data
    print("Loading datasets...")
    df_norm_list = [pd.read_parquet(p, engine='fastparquet') for p in args.normalized]
    df_norm = pd.concat(df_norm_list, ignore_index=True)
    df_pairs = pd.read_parquet(args.candidates, engine='fastparquet')

    # Map entity_id to name_norm
    norm_dict = dict(zip(df_norm['entity_id'], df_norm['name_norm']))

    # Extract names for the pairs
    print("Preparing pairs...")
    names1 = df_pairs['source1_entity_id'].map(norm_dict).fillna("").tolist()
    names2 = df_pairs['candidate_entity_id'].map(norm_dict).fillna("").tolist()

    # Load model
    print("Loading model intfloat/multilingual-e5-small...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer('intfloat/multilingual-e5-small', device=device)

    # E5 models generally perform better with the "query: " prefix for short texts
    names1_prefixed = [f"query: {n}" for n in names1]
    names2_prefixed = [f"query: {n}" for n in names2]

    # Compute embeddings
    print("Computing embeddings...")
    # Encode unique names first for massive speedup
    unique_names = list(set(names1_prefixed + names2_prefixed))
    print(f"Encoding {len(unique_names)} unique names on {device}...")
    
    embeddings = model.encode(unique_names, batch_size=args.batch_size, show_progress_bar=True, normalize_embeddings=True)
    emb_dict = {name: emb for name, emb in zip(unique_names, embeddings)}

    print("Calculating cosine similarities...")
    similarities = []
    for n1, n2, r1, r2 in zip(names1_prefixed, names2_prefixed, names1, names2):
        if not r1 or not r2:
            similarities.append(0.0)
            continue
        emb1 = emb_dict[n1]
        emb2 = emb_dict[n2]
        # Since normalize_embeddings=True, dot product is cosine similarity
        sim = float(np.dot(emb1, emb2))
        similarities.append(sim)

    # Output contract
    df_out = pd.DataFrame({
        'source1_entity_id': df_pairs['source1_entity_id'],
        'candidate_entity_id': df_pairs['candidate_entity_id'],
        'sem__embedding_cosine': similarities
    })

    print(f"Saving to {args.output}...")
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    df_out.to_parquet(args.output, index=False)
    print("Done!")

if __name__ == '__main__':
    main()
