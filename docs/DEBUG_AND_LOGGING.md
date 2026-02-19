# Debug and server console logging

## When `--debug` is set

- **Startup** (`serve.rs`): `Tofy serve: debug mode on — t/s and decoder dumps go to this console (stderr)`.
- **Per request** (`decoder_runtime.rs`): decoder stderr is piped; we print any line containing `t/s` or `tokens/s`, or a 30-line stderr/stdout fallback if none found.
- **Per request** (`world.rs`): `[tofy] response in X.XXs` (full pipeline time).

All of the above go to **stderr** of the server process (the same terminal where you run `cargo run --release -- --serve ...`). With no `--debug`, nothing is printed to the console after requests.
