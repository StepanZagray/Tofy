#![allow(clippy::module_inception)]

pub mod data;
pub mod hub;

pub use data::{
    build_vocab_from_pair_file, build_vocab_from_raw_world_file_with_mode, count_pairs_with_vocab,
    count_raw_world_rows, count_raw_world_rows_split, count_raw_world_rows_split_with_mode,
    encode_line_with_vocab_mode, encode_raw_world_line_with_vocab_mode,
    encode_text_with_vocab_mode, encode_world_examples, encode_world_examples_with_mode,
    make_augmented_jepa_batch, make_augmented_jepa_batch_from_pairs, make_decoder_batch,
    make_decoder_batch_from_slice, make_jepa_batch_from_pairs, make_world_batch_from_slice,
    tokenize_for_inference, tokenizer_spec, tokenizer_spec_signature, CachedDecoderExample,
    CachedDecoderStream, CachedPairStream, CachedWorldStream, CurriculumDenoisingConfig,
    PairStream, RawWorldExample, RawWorldStream, TokenizationMode, TokenizerSpec, WorldExample,
    ACTION_CODE, ACTION_DONE, ACTION_FETCH_DOCS, DEFAULT_MIN_TOKENS_PER_LINE,
    DEFAULT_STREAM_SHUFFLE_BUFFER,
};
pub use hub::{
    ensure_hub_dataset_cached, ensure_hub_wikipedia_cached,
    ensure_hub_wikipedia_cached_with_max_files, prepare_ultrachat_pairs,
};
