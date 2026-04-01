pub mod data;
pub mod hub;

pub use data::{
    build_vocab_from_pair_file, build_vocab_from_raw_world_file, count_pairs_with_vocab,
    count_raw_world_rows, encode_world_examples, make_decoder_batch,
    make_jepa_batch_from_pairs, make_world_batch_from_slice, tokenize_for_inference,
    PairStream, RawWorldStream, DEFAULT_MIN_TOKENS_PER_LINE, DEFAULT_STREAM_SHUFFLE_BUFFER,
};
pub use hub::{
    ensure_hub_dataset_cached, ensure_hub_wikipedia_cached, prepare_ultrachat_pairs,
};
