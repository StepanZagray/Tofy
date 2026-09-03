"""Shard schema/dtypes, manifest, public-id attestation and determinism (ADR 0005 §2.6)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tofy_arc3.synth import census, generate
from tofy_arc3.synth.public_games import PUBLIC_GAME_IDS, assert_not_public
from tofy_arc3.synth.shard import MAX_SHARD_ROWS, TENSOR_DTYPES, read_manifest, read_shard, write_manifest


@pytest.fixture(scope="module")
def sample(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("synth") / "a"
    generate.generate(seed=3, games=2, out=out, episode_steps=40, shard_rows=100, verbose=False)
    return out


def test_shard_schema_and_dtypes(sample: Path):
    m = read_manifest(sample / "manifest.json")
    assert m["source"] == "tofy_synth_arcengine"
    assert m["total_rows"] == 4 * 40
    assert len(m["shards"]) == 2  # 160 rows at 100 per shard
    for entry in m["shards"]:
        t = read_shard(sample / entry["file"])
        n = t["frames"].shape[0]
        assert n == entry["rows"] <= MAX_SHARD_ROWS
        for name, (dtype, tail) in TENSOR_DTYPES.items():
            assert t[name].dtype == dtype, name
            assert t[name].shape == (n,) + tail, name
        assert set(t) == set(TENSOR_DTYPES)
        assert t["frames"].max() <= 15 and t["next_frames"].max() <= 15
        assert hashlib.sha256((sample / entry["file"]).read_bytes()).hexdigest() == entry["sha256"]
        # actions: id in {0..6}; x,y == 255 unless ACTION6
        ids = t["actions"][:, 0]
        assert ids.max() <= 6
        non_click = ids != 6
        assert (t["actions"][non_click, 1:] == 255).all()
        assert (t["actions"][~non_click, 1:] <= 63).all()
        assert (t["available_actions"] & 1).all()  # RESET bit
        assert set(np.unique(t["level_completed"])) <= {0, 1}
        assert set(np.unique(t["game_over"])) <= {0, 1}


def test_manifest_contract_fields(sample: Path):
    m = read_manifest(sample / "manifest.json")
    assert m["generator"]["git_revision"]
    assert m["seed"] == 3
    assert m["public_game_ids_excluded"] == list(PUBLIC_GAME_IDS)
    assert m["public_game_ids_intersection"] == []
    assert m["game_id_prefix"] == "tsyn"
    assert all(g.startswith("tsyn-") for g in m["game_ids"])
    assert m["twin_pair_count"] == 2 and m["twin_pairs"] == [[0, 1], [2, 3]]
    rc = m["rule_census"]
    assert rc["distinct_rule_ids"] == 4 and rc["games"] == 4
    assert sum(rc["by_goal"].values()) == 4
    games = json.loads((sample / m["games_file"]).read_text())
    assert len(games) == 4 and all(len(g["levels"]) >= 2 for g in games)


def test_rule_id_halves_and_episode_layout(sample: Path):
    m = read_manifest(sample / "manifest.json")
    parts = [read_shard(sample / s["file"]) for s in m["shards"]]
    t = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    rid = t["rule_id_lo"].astype(np.uint64) | (t["rule_id_hi"].astype(np.uint64) << np.uint64(32))
    by_ep = {e["episode"]: e["rule_id"] for e in m["episodes"]}
    for ep in np.unique(t["episode"]):
        rows = t["episode"] == ep
        assert set(rid[rows].tolist()) == {by_ep[int(ep)]}
        ti = t["transition_index"][rows]
        assert np.array_equal(np.sort(ti), np.arange(rows.sum()))
    # chronological frame chaining within an episode: next_frames[t] == frames[t+1]
    for ep in np.unique(t["episode"]):
        idx = np.where(t["episode"] == ep)[0]
        idx = idx[np.argsort(t["transition_index"][idx])]
        assert np.array_equal(t["next_frames"][idx[:-1]], t["frames"][idx[1:]])


def test_census_single_frame_identifiable_is_zero(sample: Path):
    result = census.run_census(sample, random_steps=200)
    tw = result["twins"]
    assert tw["single_frame_rule_identifiable"] == 0
    assert tw["twin_pairs_frame0_mismatch"] == 0
    assert tw["twin_pairs"] == 2
    assert tw["first_divergence"]["pass"] == 2
    assert result["public_game_ids_intersection"] == []


def test_determinism_same_seed_identical_bytes(tmp_path: Path):
    a, b = tmp_path / "a", tmp_path / "b"
    generate.generate(seed=5, games=1, out=a, episode_steps=30, verbose=False)
    generate.generate(seed=5, games=1, out=b, episode_steps=30, verbose=False)
    for name in ("shard-00000.safetensors", "manifest.json", "games.json"):
        assert (a / name).read_bytes() == (b / name).read_bytes(), name
    c = tmp_path / "c"
    generate.generate(seed=6, games=1, out=c, episode_steps=30, verbose=False)
    assert (a / "shard-00000.safetensors").read_bytes() != (c / "shard-00000.safetensors").read_bytes()


def test_public_id_attestation_fails_closed(tmp_path: Path):
    with pytest.raises(ValueError):
        assert_not_public(["tsyn-12345678", PUBLIC_GAME_IDS[0]])
    with pytest.raises(ValueError):
        assert_not_public(["ls20-deadbeef"])
    with pytest.raises(ValueError):
        write_manifest(tmp_path / "m.json", {"source": "other"})
