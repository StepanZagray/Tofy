//! Only materialized fit/fresh prefixes are opened; source C8 paths are provenance.
use super::head::{Kind, WIDTH};
use anyhow::{ensure, Context, Result};
use candle_core::{Device, Tensor};
use clap::ValueEnum;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    collections::HashSet,
    fs::{self, File},
    io::{BufRead, BufReader, Read},
    path::Path,
};

pub const FIT_ROWS: usize = 512;
pub const FRESH_ROWS: usize = 256;
pub const PERMUTATION_SHA: &str =
    "613f5bb549e9e4a83d947d1a1b40d8bf9228b8b1418c96ca0af67ba6d4328228";

#[derive(Clone, Copy, Debug, PartialEq, ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Core {
    Initial,
    Final,
}
impl Core {
    pub fn checkpoint(self) -> &'static str {
        match self {
            Self::Initial => "4cd502f7a76fe3dd9729f6693e9d707c670972f7ebe58ad1cbdde37ea2bb2802",
            Self::Final => "a983626a888279cbb31651c6abcd1ebf4f808c8927a29a8349d03ffc35bd4e5a",
        }
    }
    fn fit_manifest(self) -> &'static str {
        match self {
            Self::Initial => "91143312730545f601da8244e56fa356d52673f3b43ec8cbd5dc9033b567d7bc",
            Self::Final => "187b72b7fde5737a70a50a3d7ab6b8d2df416baebcd85de28e2086631b0f2137",
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Row {
    pub input_index: usize,
    pub episode_id: u64,
    pub partition: String,
    pub input_sha256: String,
    pub query_sha256: String,
    pub label_sha256: String,
    pub correct_action: u32,
}

pub struct Cache {
    pub manifest: Value,
    pub rows: Vec<Row>,
    pub values: Vec<f32>,
}

pub fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
pub fn file_hash(path: &Path) -> Result<String> {
    let mut hash = Sha256::new();
    let mut file = File::open(path)?;
    let mut buffer = [0u8; 65536];
    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        hash.update(&buffer[..n]);
    }
    Ok(format!("{:x}", hash.finalize()))
}
pub fn hash_text(text: &str) -> Result<()> {
    ensure!(
        text.len() == 64
            && text
                .bytes()
                .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase()),
        "invalid SHA256"
    );
    Ok(())
}

pub fn verified_manifest(root: &Path, expected: &str, object_entries: bool) -> Result<Value> {
    ensure!(
        root.is_absolute() && root.is_dir() && !root.is_symlink(),
        "cache/head root must be an absolute regular directory"
    );
    hash_text(expected)?;
    let path = root.join("manifest.json");
    ensure!(
        path.is_file() && !path.is_symlink() && file_hash(&path)? == expected,
        "manifest SHA256 mismatch"
    );
    let manifest: Value = serde_json::from_slice(&fs::read(&path)?)?;
    let files = manifest["files"]
        .as_object()
        .context("manifest files missing")?;
    let mut actual = HashSet::new();
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        ensure!(
            entry.file_type()?.is_file(),
            "artifact root contains nonregular entry"
        );
        let name = entry
            .file_name()
            .into_string()
            .map_err(|_| anyhow::anyhow!("non-UTF8 artifact name"))?;
        if name != "manifest.json" {
            actual.insert(name);
        }
    }
    ensure!(
        actual == files.keys().cloned().collect(),
        "manifest/tree file population mismatch"
    );
    for (name, item) in files {
        ensure!(
            Path::new(name).file_name().and_then(|x| x.to_str()) == Some(name),
            "manifest filename is not local"
        );
        let (sha, bytes) = if object_entries {
            (
                item["sha256"].as_str().context("missing artifact hash")?,
                Some(item["bytes"].as_u64().context("missing byte count")?),
            )
        } else {
            (item.as_str().context("missing head artifact hash")?, None)
        };
        hash_text(sha)?;
        let path = root.join(name);
        ensure!(file_hash(&path)? == sha, "artifact digest mismatch: {name}");
        if let Some(bytes) = bytes {
            ensure!(
                path.metadata()?.len() == bytes,
                "artifact byte count mismatch: {name}"
            );
        }
    }
    Ok(manifest)
}

pub fn load(root: &Path, expected: &str, core: Core, kind: Kind, fresh: bool) -> Result<Cache> {
    load_population(root, expected, core, kind, fresh, None)
}

pub fn load_confirmation(
    root: &Path,
    expected: &str,
    core: Core,
    kind: Kind,
    panel: u8,
) -> Result<Cache> {
    ensure!(panel < 3, "confirmation panel must be 0, 1 or 2");
    load_population(root, expected, core, kind, true, Some(panel))
}

fn load_population(
    root: &Path,
    expected: &str,
    core: Core,
    kind: Kind,
    fresh: bool,
    panel: Option<u8>,
) -> Result<Cache> {
    let manifest = verified_manifest(root, expected, true)?;
    let (count, partition, seed, tag) = if let Some(index) = panel {
        (
            FRESH_ROWS,
            "confirmation_eval",
            20260917 + u64::from(index),
            0x43554441434f4e46 + u64::from(index) * 0x10000,
        )
    } else if fresh {
        (FRESH_ROWS, "fresh_eval", 20260916u64, 0x524541444f5554u64)
    } else {
        (FIT_ROWS, "fit", 20260915, 0x46454154555245)
    };
    ensure!(
        manifest["schema"]
            == if panel.is_some() {
                "looped-imported-readout-cache-v1"
            } else {
                "looped-learned-readout-cache-v1"
            }
            && manifest["partition"] == partition
            && manifest["rows"] == count
            && manifest["core_checkpoint_sha256"] == core.checkpoint(),
        "cache partition/checkpoint/schema mismatch"
    );
    let files = manifest["files"].as_object().context("cache files")?;
    ensure!(
        files.keys().map(String::as_str).collect::<HashSet<_>>()
            == ["cls.f32", "current.f32", "rows.jsonl"]
                .into_iter()
                .collect(),
        "cache must contain only materialized features and projected identities"
    );
    let source = &manifest["source"];
    if let Some(index) = panel {
        ensure!(
            source["confirmation_panel"] == index,
            "cache confirmation panel mismatch"
        );
    }
    ensure!(
        source["data_seed"] == seed
            && source["episode_id_base"] == tag
            && source["feature_schema"] == "looped-known-features-v1",
        "cache source population mismatch"
    );
    ensure!(
        Path::new(source["root"].as_str().context("source root")?).is_absolute(),
        "source root must be absolute"
    );
    for key in ["manifest_sha256", "source_revision", "binary_sha256"] {
        let value = source[key].as_str().context("source provenance missing")?;
        if key == "source_revision" {
            ensure!(
                value.len() == 40 && value.bytes().all(|x| x.is_ascii_hexdigit()),
                "bad source revision"
            );
        } else {
            hash_text(value)?;
        }
    }
    if !fresh {
        ensure!(
            source["manifest_sha256"] == core.fit_manifest()
                && source["source_revision"] == "2f2aaa711eb8f39eb823cc4e354288c7b4fbcf42"
                && source["binary_sha256"]
                    == "83150acaf7275bab63204d897972194422b2eca01137c9b7752ade456105ea8e",
            "fit cache is not bound to the registered C8 source"
        );
    }
    for (key, n) in [("cls", WIDTH), ("current", 64 * WIDTH)] {
        let bytes = count * n * 4;
        let file = format!("{key}.f32");
        let array = &manifest["source_arrays"][key];
        ensure!(
            files[&file]["bytes"] == bytes
                && array["byte_offset"] == 0
                && array["byte_length"] == bytes
                && array["file"] == format!("known-features-{key}.f32"),
            "feature source offset/shape mismatch"
        );
        hash_text(array["sha256"].as_str().context("source feature hash")?)?;
    }
    let rows = BufReader::new(File::open(root.join("rows.jsonl"))?)
        .lines()
        .map(|line| Ok(serde_json::from_str::<Row>(&line?)?))
        .collect::<Result<Vec<_>>>()?;
    ensure!(rows.len() == count, "wrong cache row count");
    let mut queries = HashSet::new();
    let mut inputs = HashSet::new();
    let mut labels = [0; 4];
    for (i, row) in rows.iter().enumerate() {
        ensure!(
            row.input_index == i
                && row.episode_id == tag + i as u64
                && row.partition == partition
                && row.correct_action < 4,
            "cache row identity/partition mismatch"
        );
        hash_text(&row.query_sha256)?;
        hash_text(&row.input_sha256)?;
        hash_text(&row.label_sha256)?;
        ensure!(
            row.label_sha256 == digest(&row.correct_action.to_le_bytes()),
            "cache action digest mismatch"
        );
        ensure!(
            queries.insert(&row.query_sha256) && inputs.insert(&row.input_sha256),
            "duplicate cached query/input"
        );
        labels[row.correct_action as usize] += 1;
    }
    ensure!(labels.iter().all(|&x| x > 0), "cache lacks a policy label");
    let filename = if kind == Kind::Spatial {
        "current.f32"
    } else {
        "cls.f32"
    };
    let mut values = Vec::new();
    // Both files contain only the allowed prefix. Validate the entire local cache,
    // while passing only this head's tensor to its computation.
    for file in ["cls.f32", "current.f32"] {
        let decoded = decode_f32(&fs::read(root.join(file))?)?;
        if file == filename {
            values = decoded;
        }
    }
    ensure!(
        values.len() == kind.shape(count).iter().product::<usize>(),
        "cache tensor shape mismatch"
    );
    Ok(Cache {
        manifest,
        rows,
        values,
    })
}

pub fn decode_f32(bytes: &[u8]) -> Result<Vec<f32>> {
    ensure!(bytes.len().is_multiple_of(4), "misaligned F32 artifact");
    let values = bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().expect("four bytes")))
        .collect::<Vec<_>>();
    ensure!(
        values.iter().all(|x| x.is_finite()),
        "nonfinite cached feature/logit"
    );
    Ok(values)
}
pub fn frozen_tensor(values: Vec<f32>, kind: Kind, rows: usize, device: &Device) -> Result<Tensor> {
    let tensor = Tensor::from_vec(values, kind.shape(rows), device)?;
    ensure!(
        !tensor.track_op(),
        "cached input unexpectedly tracks gradients"
    );
    Ok(tensor)
}
pub fn validate_permutation(indices: &[usize]) -> Result<()> {
    ensure!(
        indices.len() == FIT_ROWS
            && indices.iter().copied().collect::<HashSet<_>>() == (0..FIT_ROWS).collect(),
        "label permutation must be a complete 512-index bijection"
    );
    Ok(())
}
pub fn permutation(path: &Path, expected: &str) -> Result<Vec<usize>> {
    ensure!(
        path.is_absolute() && path.is_file() && !path.is_symlink(),
        "permutation must be an absolute regular file"
    );
    ensure!(
        expected == PERMUTATION_SHA && file_hash(path)? == expected,
        "registered label permutation digest mismatch"
    );
    let indices = serde_json::from_slice::<Vec<usize>>(&fs::read(path)?)?;
    validate_permutation(&indices)?;
    Ok(indices)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn synthetic_confirmation_cache_enforces_panel_identity_and_legacy_separation() -> Result<()> {
        // Synthetic identities/features only: never call the new episode generator.
        for index in 0..3u8 {
            let dir = super::super::tests::TestDir::new("confirmation-cache")?;
            let root = &dir.0;
            let tag = 0x43554441434f4e46 + u64::from(index) * 0x10000;
            let rows = (0..FRESH_ROWS)
                .map(|i| Row {
                    input_index: i,
                    episode_id: tag + i as u64,
                    partition: "confirmation_eval".into(),
                    input_sha256: digest(&i.to_le_bytes()),
                    query_sha256: digest(&[i.to_le_bytes().as_slice(), b"q"].concat()),
                    label_sha256: digest(&((i % 4) as u32).to_le_bytes()),
                    correct_action: (i % 4) as u32,
                })
                .map(|row| Ok(serde_json::to_string(&row)? + "\n"))
                .collect::<Result<String>>()?;
            fs::write(root.join("rows.jsonl"), rows)?;
            let mut arrays = serde_json::Map::new();
            for (name, width) in [("cls", WIDTH), ("current", 64 * WIDTH)] {
                fs::write(
                    root.join(format!("{name}.f32")),
                    vec![0u8; FRESH_ROWS * width * 4],
                )?;
                arrays.insert(name.into(),json!({"file":format!("known-features-{name}.f32"),"sha256":"3".repeat(64),"byte_offset":0,"byte_length":FRESH_ROWS*width*4}));
            }
            let mut files = serde_json::Map::new();
            for name in ["cls.f32", "current.f32", "rows.jsonl"] {
                files.insert(name.into(),json!({"sha256":file_hash(&root.join(name))?,"bytes":root.join(name).metadata()?.len()}));
            }
            let manifest = json!({"schema":"looped-imported-readout-cache-v1","partition":"confirmation_eval","rows":FRESH_ROWS,
                "core_checkpoint_sha256":Core::Initial.checkpoint(),"files":files,"source_arrays":arrays,
                "source":{"root":"/synthetic-source-not-opened","manifest_sha256":"4".repeat(64),"source_revision":"5".repeat(40),"binary_sha256":"6".repeat(64),
                    "data_seed":20260917+u64::from(index),"episode_id_base":tag,"confirmation_panel":index,"feature_schema":"looped-known-features-v1"}});
            let seal = |value: &Value| -> Result<String> {
                fs::write(root.join("manifest.json"), serde_json::to_vec(value)?)?;
                file_hash(&root.join("manifest.json"))
            };
            let hash = seal(&manifest)?;
            assert_eq!(
                load_confirmation(root, &hash, Core::Initial, Kind::Spatial, index)?
                    .rows
                    .len(),
                256
            );
            assert!(
                load_confirmation(root, &hash, Core::Initial, Kind::Spatial, (index + 1) % 3)
                    .is_err()
            );
            assert!(load_confirmation(root, &hash, Core::Final, Kind::Cls, index).is_err());
            assert!(load(root, &hash, Core::Initial, Kind::Cls, true).is_err());
            assert!(load(root, &hash, Core::Initial, Kind::Cls, false).is_err());
            for (pointer, value) in [
                ("/source/confirmation_panel", json!(3)),
                ("/source/data_seed", json!(20260916)),
                ("/source/episode_id_base", json!(tag + 1)),
                ("/partition", json!("fit")),
                ("/source_arrays/current/byte_offset", json!(4)),
                ("/rows", json!(512)),
            ] {
                let mut bad = manifest.clone();
                *bad.pointer_mut(pointer).unwrap() = value;
                assert!(
                    load_confirmation(root, &seal(&bad)?, Core::Initial, Kind::Cls, index).is_err(),
                    "accepted {pointer}"
                );
            }
        }
        Ok(())
    }

    #[test]
    fn synthetic_fit_cache_rejects_checkpoint_partition_offsets_nonfinite_and_tampering(
    ) -> Result<()> {
        let dir = super::super::tests::TestDir::new("cache")?;
        let root = &dir.0;
        let rows = (0..FIT_ROWS)
            .map(|i| Row {
                input_index: i,
                episode_id: 0x46454154555245 + i as u64,
                partition: "fit".into(),
                input_sha256: digest(&i.to_le_bytes()),
                query_sha256: digest(&[i.to_le_bytes().as_slice(), b"q"].concat()),
                label_sha256: digest(&((i % 4) as u32).to_le_bytes()),
                correct_action: (i % 4) as u32,
            })
            .map(|row| Ok(serde_json::to_string(&row)? + "\n"))
            .collect::<Result<String>>()?;
        fs::write(root.join("rows.jsonl"), rows)?;
        fs::write(root.join("cls.f32"), vec![0u8; FIT_ROWS * WIDTH * 4])?;
        fs::write(
            root.join("current.f32"),
            vec![0u8; FIT_ROWS * 64 * WIDTH * 4],
        )?;
        let mut files = serde_json::Map::new();
        for name in ["cls.f32", "current.f32", "rows.jsonl"] {
            files.insert(name.into(),json!({"sha256":file_hash(&root.join(name))?,"bytes":root.join(name).metadata()?.len()}));
        }
        let mut manifest = json!({"schema":"looped-learned-readout-cache-v1","partition":"fit","rows":FIT_ROWS,
            "core_checkpoint_sha256":Core::Initial.checkpoint(),"files":files,
            "source":{"root":"/not-opened-c8-source","manifest_sha256":Core::Initial.fit_manifest(),
            "source_revision":"2f2aaa711eb8f39eb823cc4e354288c7b4fbcf42",
            "binary_sha256":"83150acaf7275bab63204d897972194422b2eca01137c9b7752ade456105ea8e",
            "data_seed":20260915,"episode_id_base":0x46454154555245u64,"feature_schema":"looped-known-features-v1"},
            "source_arrays":{}});
        for (name, width) in [("cls", WIDTH), ("current", 64 * WIDTH)] {
            manifest["source_arrays"][name] = json!({"file":format!("known-features-{name}.f32"),"sha256":"3".repeat(64),"byte_offset":0,"byte_length":FIT_ROWS*width*4});
        }
        let seal = |manifest: &Value| -> Result<String> {
            fs::write(root.join("manifest.json"), serde_json::to_vec(manifest)?)?;
            file_hash(&root.join("manifest.json"))
        };
        let hash = seal(&manifest)?;
        assert_eq!(
            load(root, &hash, Core::Initial, Kind::Cls, false)?
                .rows
                .len(),
            FIT_ROWS
        );
        assert!(load(root, &hash, Core::Final, Kind::Cls, false).is_err());
        assert!(load(root, &hash, Core::Initial, Kind::Cls, true).is_err());
        assert!(load(root, &"0".repeat(64), Core::Initial, Kind::Cls, false).is_err());
        manifest["source_arrays"]["current"]["byte_offset"] = json!(4);
        assert!(load(root, &seal(&manifest)?, Core::Initial, Kind::Cls, false).is_err());
        manifest["source_arrays"]["current"]["byte_offset"] = json!(0);
        // The unused local feature file must also be finite, even with valid hashes.
        let mut bytes = fs::read(root.join("current.f32"))?;
        bytes[..4].copy_from_slice(&f32::NAN.to_le_bytes());
        fs::write(root.join("current.f32"), bytes)?;
        manifest["files"]["current.f32"]["sha256"] = json!(file_hash(&root.join("current.f32"))?);
        assert!(load(root, &seal(&manifest)?, Core::Initial, Kind::Cls, false).is_err());
        fs::write(root.join("extra-eval-labels.json"), b"[]")?;
        assert!(verified_manifest(root, &seal(&manifest)?, true).is_err());
        Ok(())
    }
    #[test]
    fn permutation_and_raw_binary_fail_closed() -> Result<()> {
        let valid = (0..FIT_ROWS).rev().collect::<Vec<_>>();
        validate_permutation(&valid)?;
        let mut duplicate = valid.clone();
        duplicate[0] = duplicate[1];
        assert!(validate_permutation(&duplicate).is_err());
        assert!(validate_permutation(&valid[..511]).is_err());
        let mut outside = valid;
        outside[0] = 512;
        assert!(validate_permutation(&outside).is_err());
        let values = [0.0f32, -1.25, 3.5];
        let bytes = values
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect::<Vec<_>>();
        assert_eq!(decode_f32(&bytes)?, values);
        assert!(decode_f32(&bytes[..11]).is_err());
        assert!(decode_f32(&f32::NAN.to_le_bytes()).is_err());
        assert!(decode_f32(&f32::INFINITY.to_le_bytes()).is_err());
        Ok(())
    }
    #[test]
    fn row_parser_forbids_privileged_fields_and_wrong_label_types() {
        let text = r#"{"input_index":0,"episode_id":1,"partition":"fit","input_sha256":"x","query_sha256":"y","label_sha256":"z","correct_action":0}"#;
        assert!(serde_json::from_str::<Row>(text).is_ok());
        for key in [
            "visible_cells",
            "agent_index",
            "goal_index",
            "coordinates",
            "target_cells",
        ] {
            let mut row: Value = serde_json::from_str(text).unwrap();
            row[key] = serde_json::json!([1, 2]);
            assert!(serde_json::from_value::<Row>(row).is_err());
        }
        assert!(serde_json::from_str::<Row>(
            &text.replace("\"correct_action\":0", "\"correct_action\":0.5")
        )
        .is_err());
    }
}
