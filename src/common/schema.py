# src/common/schema.py

NORMALIZED_COLUMNS = [
    'entity_id',
    'country',
    'name_raw',
    'name_norm',
    'name_core',
    'legal_suffix',
    'name_tokens',
    'name_transliterated',
    'is_non_latin',
    'address_raw',
    'address_norm',
    'address_numbers',
    'address_pin'
]

CANDIDATE_PAIRS_COLUMNS_TRAIN = [
    'source1_entity_id',
    'candidate_entity_id',
    'candidate_source',
    'country',
    'name_tfidf_block_hit',
    'char_ngram_block_hit',
    'address_block_hit',
    'exact_key_hit',
    'pin_block_hit',
    'phonetic_block_hit',
    'embedding_block_hit',
    'bidirectional_block_hit',
    'num_legs_retrieved',
    'blocking_score',
    'is_true_match'
]

CANDIDATE_PAIRS_COLUMNS_TEST = CANDIDATE_PAIRS_COLUMNS_TRAIN[:-1]
