//! C11 imports fixed parameters with their actual producer recipe, never a fake fit.
use super::{
    cache,
    head::{self, Kind},
    Core,
};
use anyhow::{ensure, Context, Result};
use candle_core::{Device, Tensor};
use candle_nn::VarMap;
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, HashSet},
    fs,
    path::{Path, PathBuf},
};

pub const SCHEMA: &str = "looped-imported-readout-evaluation-v1";

pub fn gemm_reduced_precision(device: &str) -> Option<bool> {
    #[cfg(feature = "cudnn")]
    if device.starts_with("cuda") {
        return Some(candle_core::cuda_backend::gemm_reduced_precision_f32());
    }
    let _ = device;
    None
}
const PARENTS: [&str; 3] = [
    "ebae6f94562879ed343895dc09e9243ab616900f5f8f5e64b54f7ac408cf9e60",
    "ea0fb1e773400363e52e3b6966600fde5d8e0ffe37a14547e49772d2186fe477",
    "76565de6508b903ff19537a7f63352f1d649debef89abf1773b93096f8b727e8",
];

#[derive(Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum SourceKind {
    C10RoleRidge,
    C9Adamw,
}
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Producer {
    source_revision: String,
    implementation_sha256: String,
}
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Recipe {
    name: String,
    original_optimizer_updates: usize,
    privileged_role_supervision: bool,
    cast: String,
}
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Parents {
    c8: String,
    c9: String,
    c10: String,
}
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Original {
    path: PathBuf,
    sha256: String,
}
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Artifact {
    file: String,
    sha256: String,
    bytes: usize,
}
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Section {
    name: String,
    dtype: String,
    shape: Vec<usize>,
    byte_offset: usize,
    byte_length: usize,
    sha256: String,
}
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    schema: String,
    pub source_kind: SourceKind,
    head_kind: Kind,
    core_checkpoint: Core,
    core_checkpoint_sha256: String,
    pub arm: String,
    producer: Producer,
    recipe: Recipe,
    parent_manifests: Parents,
    original_artifact: Original,
    artifact: Artifact,
    tensors: Vec<Section>,
}

fn shapes(kind: Kind) -> Vec<(&'static str, Vec<usize>)> {
    match kind {
        Kind::Spatial => vec![
            ("queries", vec![2, 128]),
            ("output.weight", vec![4, 256]),
            ("output.bias", vec![4]),
        ],
        Kind::Cls => vec![
            ("hidden.weight", vec![10, 128]),
            ("hidden.bias", vec![10]),
            ("output.weight", vec![4, 10]),
            ("output.bias", vec![4]),
        ],
    }
}

impl Manifest {
    fn validate(&self, core: Core, kind: Kind) -> Result<()> {
        ensure!(
            self.schema == "looped-imported-readout-source-v1"
                && self.core_checkpoint == core
                && self.core_checkpoint_sha256 == core.checkpoint()
                && self.head_kind == kind,
            "import schema/core/head mismatch"
        );
        let ridge = self.source_kind == SourceKind::C10RoleRidge;
        ensure!(
            match self.arm.as_str() {
                "c10_true" | "c10_null" => ridge && kind == Kind::Spatial,
                "c9_spatial" | "c9_null" => !ridge && kind == Kind::Spatial,
                "c9_cls" => !ridge && kind == Kind::Cls,
                _ => false,
            },
            "import source/arm/kind mismatch"
        );
        let (revision, implementation, recipe, updates, cast) = if ridge {
            (
                "82ac8cb6ea3a06ae13d836d526c47994cd675d18",
                "a49151b2cd1294aa86fc72433755feb92e3d6b855796e8ee157e2402b51ac748",
                "c10_role_ridge_c8_affine_v1",
                0,
                "f64_to_f32_once_no_rescale",
            )
        } else {
            (
                "06a8d76fada203c5b1a11a45782ed777df27ff4c",
                "fecbc1b91099ddc9b39ac9b6361ee8072c26563f6b710e06c29ec7b3adeef20e",
                "c9_adamw1000_v1",
                1000,
                "identity_f32",
            )
        };
        ensure!(
            self.producer.source_revision == revision
                && self.producer.implementation_sha256 == implementation
                && self.recipe.name == recipe
                && self.recipe.original_optimizer_updates == updates
                && self.recipe.privileged_role_supervision == ridge
                && self.recipe.cast == cast,
            "import producer or original recipe mismatch"
        );
        // These are confirmation-context seals, not claims that C10 preceded C9.
        ensure!(
            [
                self.parent_manifests.c8.as_str(),
                self.parent_manifests.c9.as_str(),
                self.parent_manifests.c10.as_str()
            ] == PARENTS,
            "import parent context mismatch"
        );
        ensure!(
            self.artifact.file == "parameters.f32" && self.artifact.bytes == kind.parameters() * 4,
            "wrong canonical parameter artifact"
        );
        cache::hash_text(&self.artifact.sha256)?;
        cache::hash_text(&self.original_artifact.sha256)?;
        let expected = shapes(kind);
        ensure!(
            self.tensors.len() == expected.len(),
            "wrong imported tensor count"
        );
        let mut offset = 0;
        for (section, (name, shape)) in self.tensors.iter().zip(expected) {
            let bytes = shape.iter().product::<usize>() * 4;
            ensure!(
                section.name == name
                    && section.dtype == "F32LE"
                    && section.shape == shape
                    && section.byte_offset == offset
                    && section.byte_length == bytes,
                "noncanonical tensor name/shape/dtype/offset: {name}"
            );
            cache::hash_text(&section.sha256)?;
            offset += bytes;
        }
        ensure!(
            offset == self.artifact.bytes,
            "parameter payload length mismatch"
        );
        Ok(())
    }
}

pub struct Imported {
    pub manifest: Manifest,
    bytes: Vec<u8>,
}
impl Imported {
    pub fn load(root: &Path, expected: &str, core: Core, kind: Kind) -> Result<Self> {
        ensure!(
            root.is_absolute() && root.is_dir() && !root.is_symlink(),
            "import root must be an absolute regular directory"
        );
        cache::hash_text(expected)?;
        let mut files = HashSet::new();
        for entry in fs::read_dir(root)? {
            let entry = entry?;
            ensure!(entry.file_type()?.is_file(), "nonregular import entry");
            files.insert(entry.file_name());
        }
        ensure!(
            files
                == ["manifest.json", "parameters.f32"]
                    .into_iter()
                    .map(std::ffi::OsString::from)
                    .collect(),
            "unexpected import files"
        );
        let path = root.join("manifest.json");
        let manifest_bytes = fs::read(path)?;
        ensure!(
            cache::digest(&manifest_bytes) == expected,
            "import manifest hash mismatch"
        );
        let manifest: Manifest = serde_json::from_slice(&manifest_bytes)?;
        manifest.validate(core, kind)?;
        let original = &manifest.original_artifact;
        ensure!(
            original.path.is_absolute()
                && original.path.is_file()
                && !original.path.is_symlink()
                && cache::file_hash(&original.path)? == original.sha256,
            "original source artifact mismatch"
        );
        let bytes = fs::read(root.join("parameters.f32"))?;
        ensure!(
            bytes.len() == manifest.artifact.bytes
                && cache::digest(&bytes) == manifest.artifact.sha256,
            "import payload mismatch"
        );
        cache::decode_f32(&bytes)?;
        for section in &manifest.tensors {
            ensure!(
                cache::digest(
                    &bytes[section.byte_offset..section.byte_offset + section.byte_length]
                ) == section.sha256,
                "import tensor hash mismatch: {}",
                section.name
            );
        }
        Ok(Self { manifest, bytes })
    }
    pub fn apply(&self, vars: &VarMap, device: &Device) -> Result<()> {
        let named = vars
            .data()
            .lock()
            .map_err(|_| anyhow::anyhow!("parameter lock poisoned"))?;
        ensure!(
            named.len() == self.manifest.tensors.len(),
            "wrong destination tensor set"
        );
        for section in &self.manifest.tensors {
            let var = named
                .get(&section.name)
                .context("missing destination tensor")?;
            ensure!(
                var.dims() == section.shape,
                "destination tensor shape mismatch"
            );
            let values = cache::decode_f32(
                &self.bytes[section.byte_offset..section.byte_offset + section.byte_length],
            )?;
            var.set(&Tensor::from_vec(values, section.shape.clone(), device)?)?;
        }
        drop(named);
        self.verify(vars)
    }
    fn verify_tensors(&self, tensors: &HashMap<String, Tensor>) -> Result<()> {
        ensure!(
            tensors.len() == self.manifest.tensors.len(),
            "wrong roundtrip tensor set"
        );
        for section in &self.manifest.tensors {
            let tensor = tensors
                .get(&section.name)
                .context("missing roundtrip tensor")?;
            ensure!(
                tensor.dtype() == candle_core::DType::F32 && tensor.dims() == section.shape,
                "roundtrip tensor dtype/shape mismatch"
            );
            let bytes = super::finite(tensor)?
                .iter()
                .flat_map(|x| x.to_le_bytes())
                .collect::<Vec<_>>();
            ensure!(
                bytes == self.bytes[section.byte_offset..section.byte_offset + section.byte_length],
                "canonical imported tensor changed: {}",
                section.name
            );
        }
        Ok(())
    }
    pub fn verify(&self, vars: &VarMap) -> Result<()> {
        self.verify_tensors(
            &head::named(vars)
                .into_iter()
                .map(|(name, var)| (name, var.as_tensor().clone()))
                .collect(),
        )
    }
    pub fn verify_saved(&self, path: &Path) -> Result<()> {
        self.verify_tensors(&candle_core::safetensors::load(path, &Device::Cpu)?)
    }
}

#[cfg(test)]
mod tests {
    use super::super::{head::Head, tests::TestDir};
    use super::*;
    use candle_core::DType;
    use candle_nn::VarBuilder;
    use serde_json::{json, Value};

    struct Fixture {
        _dir: TestDir,
        root: PathBuf,
        manifest: Value,
    }
    impl Fixture {
        fn new(kind: Kind, ridge: bool) -> Result<Self> {
            let dir = TestDir::new("import")?;
            let root = dir.0.join("sealed");
            fs::create_dir(&root)?;
            let original = dir.0.join("synthetic-parent");
            fs::write(&original, b"synthetic parent, not real evidence")?;
            let mut bytes = Vec::new();
            let mut sections = Vec::new();
            for (name, shape) in shapes(kind) {
                let offset = bytes.len();
                for i in 0..shape.iter().product::<usize>() {
                    let x = if i == 0 {
                        -0.0
                    } else {
                        ((i % 19) as f32 - 9.0) / 32.0
                    };
                    bytes.extend_from_slice(&x.to_le_bytes());
                }
                sections.push(
                    json!({"name":name,"dtype":"F32LE","shape":shape,"byte_offset":offset,
                    "byte_length":bytes.len()-offset,"sha256":cache::digest(&bytes[offset..])}),
                );
            }
            fs::write(root.join("parameters.f32"), &bytes)?;
            let manifest = json!({"schema":"looped-imported-readout-source-v1",
                "source_kind":if ridge {"c10_role_ridge"} else {"c9_adamw"},"head_kind":kind,
                "core_checkpoint":"initial","core_checkpoint_sha256":Core::Initial.checkpoint(),
                "arm":if ridge {"c10_true"} else if kind==Kind::Spatial {"c9_spatial"} else {"c9_cls"},
                "producer":{"source_revision":if ridge {"82ac8cb6ea3a06ae13d836d526c47994cd675d18"} else {"06a8d76fada203c5b1a11a45782ed777df27ff4c"},
                    "implementation_sha256":if ridge {"a49151b2cd1294aa86fc72433755feb92e3d6b855796e8ee157e2402b51ac748"} else {"fecbc1b91099ddc9b39ac9b6361ee8072c26563f6b710e06c29ec7b3adeef20e"}},
                "recipe":{"name":if ridge {"c10_role_ridge_c8_affine_v1"} else {"c9_adamw1000_v1"},
                    "original_optimizer_updates":if ridge {0} else {1000},"privileged_role_supervision":ridge,
                    "cast":if ridge {"f64_to_f32_once_no_rescale"} else {"identity_f32"}},
                "parent_manifests":{"c8":PARENTS[0],"c9":PARENTS[1],"c10":PARENTS[2]},
                "original_artifact":{"path":original,"sha256":cache::file_hash(&original)?},
                "artifact":{"file":"parameters.f32","bytes":bytes.len(),"sha256":cache::digest(&bytes)},"tensors":sections});
            Ok(Self {
                _dir: dir,
                root,
                manifest,
            })
        }
        fn load(&self, manifest: &Value, core: Core, kind: Kind) -> Result<Imported> {
            let bytes = serde_json::to_vec(manifest)?;
            fs::write(self.root.join("manifest.json"), &bytes)?;
            Imported::load(&self.root, &cache::digest(&bytes), core, kind)
        }
    }

    #[test]
    fn canonical_import_roundtrips_spatial_and_cls_and_exports_actual_outputs() -> Result<()> {
        for (kind, ridge) in [
            (Kind::Spatial, true),
            (Kind::Spatial, false),
            (Kind::Cls, false),
        ] {
            let fixture = Fixture::new(kind, ridge)?;
            let source = fixture.load(&fixture.manifest, Core::Initial, kind)?;
            let vars = VarMap::new();
            let head = Head::new(
                kind,
                VarBuilder::from_varmap(&vars, DType::F32, &Device::Cpu),
            )?;
            source.apply(&vars, &Device::Cpu)?;
            let saved = fixture._dir.0.join("head.safetensors");
            vars.save(&saved)?;
            source.verify_saved(&saved)?;
            let h = cache::frozen_tensor(
                (0..kind.shape(2).iter().product())
                    .map(|i| (i % 31) as f32 / 31.0)
                    .collect(),
                kind,
                2,
                &Device::Cpu,
            )?;
            let output = head.forward(&h)?;
            source.verify(&vars)?;
            assert!(!h.track_op());
            let rows = (0..2)
                .map(|i| cache::Row {
                    input_index: i,
                    episode_id: i as u64,
                    partition: "synthetic".into(),
                    input_sha256: "a".repeat(64),
                    query_sha256: "b".repeat(64),
                    label_sha256: cache::digest(&(i as u32).to_le_bytes()),
                    correct_action: i as u32,
                })
                .collect::<Vec<_>>();
            let outdir = fixture._dir.0.join("output");
            fs::create_dir(&outdir)?;
            super::super::write_predictions(&outdir, &rows, &[0, 1], &output, true)?;
            let records = fs::read_to_string(outdir.join("predictions.jsonl"))?
                .lines()
                .map(serde_json::from_str::<Value>)
                .collect::<std::result::Result<Vec<_>, _>>()?;
            for (name, tensor) in [
                ("pooled", Some(&output.pooled)),
                ("attention", output.attention.as_ref()),
            ] {
                if let Some(tensor) = tensor {
                    let raw = fs::read(outdir.join(format!("{name}.f32")))?;
                    assert_eq!(cache::decode_f32(&raw)?, super::super::finite(tensor)?);
                    for (i, row) in records.iter().enumerate() {
                        let len = tensor.elem_count() / 2 * 4;
                        assert_eq!(row[name]["byte_offset"], i * len);
                        assert_eq!(row[name]["byte_length"], len);
                    }
                } else {
                    assert!(records[0][name].is_null());
                    assert!(!outdir.join(format!("{name}.f32")).exists());
                }
            }
            let mut changed = VarMap::new();
            let _ = Head::new(
                kind,
                VarBuilder::from_varmap(&changed, DType::F32, &Device::Cpu),
            )?;
            changed.load(saved)?;
            source.verify(&changed)?;
            head::named(&changed)[0]
                .1
                .set(&head::named(&changed)[0].1.affine(1.0, 1.0)?)?;
            assert!(source.verify(&changed).is_err());
        }
        Ok(())
    }

    #[test]
    fn import_rejects_mixed_producers_recipes_offsets_types_and_nonfinite_payloads() -> Result<()> {
        let fixture = Fixture::new(Kind::Spatial, true)?;
        fixture.load(&fixture.manifest, Core::Initial, Kind::Spatial)?;
        assert!(fixture
            .load(&fixture.manifest, Core::Final, Kind::Spatial)
            .is_err());
        assert!(fixture
            .load(&fixture.manifest, Core::Initial, Kind::Cls)
            .is_err());
        for (pointer, value) in [
            ("/source_kind", json!("c9_adamw")),
            ("/arm", json!("c9_spatial")),
            ("/producer/source_revision", json!("0".repeat(40))),
            ("/producer/implementation_sha256", json!("0".repeat(64))),
            ("/recipe/original_optimizer_updates", json!(1000)),
            ("/recipe/privileged_role_supervision", json!(false)),
            ("/recipe/cast", json!("identity_f32")),
            ("/parent_manifests/c10", json!("0".repeat(64))),
            ("/tensors/1/byte_offset", json!(0)),
            ("/tensors/0/shape", json!([128, 2])),
            ("/tensors/0/dtype", json!("F64LE")),
            ("/tensors/0/sha256", json!("0".repeat(64))),
            ("/artifact/bytes", json!(5137)),
            ("/original_artifact/sha256", json!("0".repeat(64))),
        ] {
            let mut bad = fixture.manifest.clone();
            *bad.pointer_mut(pointer).unwrap() = value;
            assert!(
                fixture.load(&bad, Core::Initial, Kind::Spatial).is_err(),
                "accepted {pointer}"
            );
        }
        let mut unknown = fixture.manifest.clone();
        unknown["optimizer"] = json!("AdamW");
        assert!(fixture
            .load(&unknown, Core::Initial, Kind::Spatial)
            .is_err());
        let mut bytes = fs::read(fixture.root.join("parameters.f32"))?;
        bytes[..4].copy_from_slice(&f32::NAN.to_le_bytes());
        fs::write(fixture.root.join("parameters.f32"), &bytes)?;
        let mut bad = fixture.manifest.clone();
        bad["artifact"]["sha256"] = json!(cache::digest(&bytes));
        bad["tensors"][0]["sha256"] = json!(cache::digest(&bytes[..1024]));
        assert!(fixture.load(&bad, Core::Initial, Kind::Spatial).is_err());
        Ok(())
    }
}
