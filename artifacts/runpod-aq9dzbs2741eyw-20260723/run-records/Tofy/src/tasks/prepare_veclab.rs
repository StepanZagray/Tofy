//! Deterministic veclab corpus generator (see docs/VECLAB_DATA_SPEC.md).

use anyhow::{bail, Context, Result};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use rayon::prelude::*;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use crate::tasks::prepare::escape_pair_field;

pub const GENERATOR_VERSION: &str = "1.2.0";
pub const DEFAULT_SEED: u64 = 20260705;
pub const FUNCTION_COUNT: usize = 200;
pub const SEEN_FUNCTION_MAX: usize = 100;
pub const TASKS_PER_FUNCTION: usize = 40;
pub const EVAL_TASKS_PER_FUNCTION: usize = 3;
pub const KNOWLEDGE_ROWS_PER_FUNCTION: usize = 40;
pub const WORLD_BATCH: usize = 32;
pub const MODULE_PATH: &str = "veclab.dev/veclab";

const SYLLABLES: &[&str] = &[
    "vor", "bel", "sken", "ith", "dram", "quel", "mox", "tren", "plix", "nurb", "keth", "yul",
    "zarn", "flep", "grol", "wex", "brin", "dax", "hurn", "vix", "ombr", "stel", "pran", "kiv",
    "lor", "mex", "neth", "orv", "pax", "quen", "rilm", "sorv", "telm", "ulv", "vex", "welm",
    "xarn", "yeth", "zilm",
];

const BLOCKED_NAMES: &[&str] = &[
    "sort", "sum", "mean", "max", "min", "len", "main", "test", "true", "false", "nil", "int",
    "float", "string", "bool", "error", "print", "range", "map", "make", "new", "append",
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Family {
    SliceTransform,
    Reduction,
    Pairwise,
    Windowed,
    Filter,
    IndexSelect,
    StringEncode,
    NumParse,
    Accumulator,
    Hybrid,
}

impl Family {
    fn from_fn_id(id: usize) -> Self {
        match (id - 1) / 20 {
            0 => Self::SliceTransform,
            1 => Self::Reduction,
            2 => Self::Pairwise,
            3 => Self::Windowed,
            4 => Self::Filter,
            5 => Self::IndexSelect,
            6 => Self::StringEncode,
            7 => Self::NumParse,
            8 => Self::Accumulator,
            9 => Self::Hybrid,
            _ => unreachable!(),
        }
    }
}

#[derive(Clone, Debug)]
#[allow(dead_code)]
enum Prim {
    SortAbsDesc,
    TakeFirst(usize),
    AlternatingSum,
    ScaleInvK(usize),
    Sum,
    Product,
    Mean,
    MaxAbs,
    MinAbs,
    WindowSum(usize),
    CountAbove(f64),
    DotPair,
    Clip(f64, f64),
    AbsValues,
    Reverse,
    Offset(f64),
    Scale(f64),
    CharCodeSum,
    ParseFieldsSum,
    FoldPositive,
}

#[derive(Clone, Copy, Debug)]
enum SigKind {
    XsK,
    Xs,
    XsYs,
    XsW,
    SK,
}

#[derive(Clone, Debug)]
struct Constants {
    k: usize,
    w: usize,
    thresh: f64,
    offset: f64,
    scale: f64,
    clip_lo: f64,
    clip_hi: f64,
}

impl Constants {
    fn from_id(id: usize) -> Self {
        Self {
            k: 2 + (id % 5),
            w: 2 + (id % 4),
            thresh: ((id % 13) as f64) * 0.25,
            offset: ((id % 11) as f64) - 5.0,
            scale: 0.5 + ((id % 7) as f64) * 0.1,
            clip_lo: -2.0 - (id % 3) as f64,
            clip_hi: 2.0 + (id % 4) as f64,
        }
    }
}

#[derive(Clone, Debug)]
#[allow(dead_code)]
struct FunctionDef {
    id: usize,
    name: String,
    family: Family,
    sig: SigKind,
    pipeline: Vec<Prim>,
    constants: Constants,
}

fn recipe(family: Family, slot: usize, c: &Constants) -> Vec<Prim> {
    let k = c.k;
    let w = c.w;
    let th = c.thresh;
    match (family, slot % 20) {
        (Family::SliceTransform, 0) => vec![
            Prim::SortAbsDesc,
            Prim::TakeFirst(k),
            Prim::AlternatingSum,
            Prim::ScaleInvK(k),
        ],
        (Family::SliceTransform, 1) => vec![
            Prim::SortAbsDesc,
            Prim::TakeFirst(k),
            Prim::Sum,
            Prim::ScaleInvK(k),
        ],
        (Family::SliceTransform, 2) => vec![Prim::AbsValues, Prim::Reverse, Prim::AlternatingSum],
        (Family::SliceTransform, 3) => vec![
            Prim::Clip(c.clip_lo, c.clip_hi),
            Prim::SortAbsDesc,
            Prim::Mean,
        ],
        (Family::SliceTransform, 4) => vec![
            Prim::SortAbsDesc,
            Prim::TakeFirst(k),
            Prim::Product,
            Prim::Scale(c.scale),
        ],
        (Family::SliceTransform, n) => vec![
            Prim::Offset(c.offset),
            Prim::SortAbsDesc,
            Prim::TakeFirst(k.min(3 + n % 3)),
            if n % 2 == 0 {
                Prim::AlternatingSum
            } else {
                Prim::Sum
            },
        ],

        (Family::Reduction, 0) => vec![Prim::Sum, Prim::Scale(c.scale)],
        (Family::Reduction, 1) => vec![Prim::Product, Prim::Offset(c.offset)],
        (Family::Reduction, 2) => vec![Prim::Mean, Prim::Scale(c.scale)],
        (Family::Reduction, 3) => vec![Prim::MaxAbs],
        (Family::Reduction, 4) => vec![Prim::MinAbs],
        (Family::Reduction, n) => vec![
            Prim::AbsValues,
            if n % 3 == 0 {
                Prim::AlternatingSum
            } else if n % 3 == 1 {
                Prim::Sum
            } else {
                Prim::Product
            },
            Prim::Offset(c.offset),
        ],

        (Family::Pairwise, 0) => vec![Prim::DotPair, Prim::Scale(c.scale)],
        (Family::Pairwise, 1) => vec![Prim::DotPair, Prim::Offset(c.offset)],
        (Family::Pairwise, n) => vec![
            Prim::DotPair,
            if n % 2 == 0 {
                Prim::Scale(c.scale)
            } else {
                Prim::Offset(c.offset)
            },
        ],

        (Family::Windowed, 0) => vec![Prim::WindowSum(w), Prim::Scale(c.scale)],
        (Family::Windowed, 1) => vec![Prim::WindowSum(w), Prim::Offset(c.offset)],
        (Family::Windowed, n) => vec![
            Prim::WindowSum(w.min(k)),
            if n % 2 == 0 {
                Prim::Scale(c.scale)
            } else {
                Prim::Mean
            },
        ],

        (Family::Filter, 0) => vec![Prim::CountAbove(th), Prim::Scale(c.scale)],
        (Family::Filter, 1) => vec![Prim::CountAbove(th), Prim::Offset(c.offset)],
        (Family::Filter, n) => vec![
            Prim::CountAbove(th.max(0.1)),
            if n % 2 == 0 {
                Prim::Scale(c.scale)
            } else {
                Prim::Offset(c.offset)
            },
        ],

        (Family::IndexSelect, 0) => {
            vec![Prim::SortAbsDesc, Prim::TakeFirst(k), Prim::AlternatingSum]
        }
        (Family::IndexSelect, 1) => vec![Prim::SortAbsDesc, Prim::TakeFirst(k), Prim::Sum],
        (Family::IndexSelect, n) => vec![
            Prim::Reverse,
            Prim::TakeFirst(k),
            if n % 2 == 0 {
                Prim::AlternatingSum
            } else {
                Prim::Sum
            },
        ],

        (Family::StringEncode, 0) => vec![Prim::CharCodeSum, Prim::Scale(c.scale)],
        (Family::StringEncode, n) => vec![
            Prim::CharCodeSum,
            if n % 2 == 0 {
                Prim::Scale(c.scale)
            } else {
                Prim::Offset(c.offset)
            },
        ],

        (Family::NumParse, 0) => vec![Prim::ParseFieldsSum, Prim::Scale(c.scale)],
        (Family::NumParse, n) => vec![
            Prim::ParseFieldsSum,
            if n % 2 == 0 {
                Prim::Scale(c.scale)
            } else {
                Prim::Offset(c.offset)
            },
        ],

        (Family::Accumulator, 0) => vec![Prim::Sum, Prim::Scale(c.scale)],
        (Family::Accumulator, 1) => vec![Prim::Mean, Prim::Offset(c.offset)],
        (Family::Accumulator, n) => vec![
            Prim::AbsValues,
            if n % 2 == 0 { Prim::Sum } else { Prim::Mean },
            if n % 3 == 0 {
                Prim::Scale(c.scale)
            } else {
                Prim::Offset(c.offset)
            },
        ],

        (Family::Hybrid, 0) => vec![
            Prim::Clip(c.clip_lo, c.clip_hi),
            Prim::SortAbsDesc,
            Prim::TakeFirst(k),
            Prim::AlternatingSum,
        ],
        (Family::Hybrid, 1) => vec![
            Prim::Offset(c.offset),
            Prim::WindowSum(w),
            Prim::Scale(c.scale),
        ],
        (Family::Hybrid, n) => vec![
            Prim::AbsValues,
            Prim::TakeFirst(k),
            if n % 2 == 0 {
                Prim::AlternatingSum
            } else {
                Prim::Sum
            },
            Prim::ScaleInvK(k.max(1)),
        ],
    }
}

fn sig_for_family(family: Family) -> SigKind {
    match family {
        Family::SliceTransform | Family::IndexSelect | Family::Hybrid => SigKind::XsK,
        Family::Reduction | Family::Accumulator => SigKind::Xs,
        Family::Pairwise => SigKind::XsYs,
        Family::Windowed | Family::Filter => SigKind::XsW,
        Family::StringEncode | Family::NumParse => SigKind::SK,
    }
}

fn generate_name(rng: &mut StdRng, used: &mut HashSet<String>) -> Result<String> {
    for _ in 0..10_000 {
        let syllables = 2 + (rng.random::<u8>() % 2) as usize;
        let mut name = String::new();
        for _ in 0..syllables {
            let s = SYLLABLES[rng.random_range(0..SYLLABLES.len())];
            if name.is_empty() {
                let mut chars = s.chars();
                if let Some(c) = chars.next() {
                    name.push(c.to_ascii_uppercase());
                    name.extend(chars);
                }
            } else {
                name.push_str(s);
            }
        }
        let lower = name.to_ascii_lowercase();
        if BLOCKED_NAMES.iter().any(|w| lower.contains(w)) {
            continue;
        }
        if used.insert(name.clone()) {
            return Ok(name);
        }
    }
    bail!("failed to generate unique veclab function name");
}

fn build_functions(seed: u64) -> Result<Vec<FunctionDef>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut used = HashSet::new();
    let mut out = Vec::with_capacity(FUNCTION_COUNT);
    for id in 1..=FUNCTION_COUNT {
        let family = Family::from_fn_id(id);
        let slot = (id - 1) % 20;
        let constants = Constants::from_id(id);
        let pipeline = recipe(family, slot, &constants);
        out.push(FunctionDef {
            id,
            name: generate_name(&mut rng, &mut used)?,
            family,
            sig: sig_for_family(family),
            pipeline,
            constants,
        });
    }
    Ok(out)
}

fn sort_abs_desc(xs: &mut [f64]) {
    xs.sort_by(|a, b| {
        b.abs()
            .partial_cmp(&a.abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
}

fn alternating_sum(xs: &[f64]) -> f64 {
    let mut sign = 1.0;
    let mut total = 0.0;
    for &v in xs {
        total += sign * v;
        sign = -sign;
    }
    total
}

fn eval_slice_pipeline(pipeline: &[Prim], xs: &[f64], _k: i64, ys: Option<&[f64]>) -> f64 {
    let mut vals = xs.to_vec();
    let mut scalar = 0.0f64;
    let mut have_scalar = false;
    for prim in pipeline {
        match prim {
            Prim::SortAbsDesc => sort_abs_desc(&mut vals),
            Prim::TakeFirst(n) => {
                let lim = (*n).min(vals.len());
                vals.truncate(lim);
            }
            Prim::AlternatingSum => {
                scalar = alternating_sum(&vals);
                have_scalar = true;
            }
            Prim::ScaleInvK(n) => {
                let denom = (*n).max(1) as f64;
                scalar = if have_scalar {
                    scalar / denom
                } else {
                    alternating_sum(&vals) / denom
                };
                have_scalar = true;
            }
            Prim::Sum => {
                scalar = vals.iter().sum();
                have_scalar = true;
            }
            Prim::Product => {
                scalar = vals.iter().product();
                have_scalar = true;
            }
            Prim::Mean => {
                scalar = if vals.is_empty() {
                    0.0
                } else {
                    vals.iter().sum::<f64>() / vals.len() as f64
                };
                have_scalar = true;
            }
            Prim::MaxAbs => return vals.iter().map(|v| v.abs()).fold(0.0, f64::max),
            Prim::MinAbs => {
                if vals.is_empty() {
                    return 0.0;
                }
                return vals.iter().map(|v| v.abs()).fold(f64::INFINITY, f64::min);
            }
            Prim::WindowSum(w) => {
                let w = (*w).max(1).min(vals.len().max(1));
                scalar = vals.iter().take(w).sum();
                have_scalar = true;
            }
            Prim::CountAbove(th) => {
                scalar = vals.iter().filter(|v| **v > *th).count() as f64;
                have_scalar = true;
            }
            Prim::DotPair => {
                let ys = ys.unwrap_or(xs);
                let n = vals.len().min(ys.len());
                scalar = (0..n).map(|i| vals[i] * ys[i]).sum();
                have_scalar = true;
            }
            Prim::Clip(lo, hi) => {
                for v in &mut vals {
                    *v = v.clamp(*lo, *hi);
                }
            }
            Prim::AbsValues => {
                for v in &mut vals {
                    *v = v.abs();
                }
            }
            Prim::Reverse => vals.reverse(),
            Prim::Offset(d) => {
                scalar = if have_scalar {
                    scalar + d
                } else {
                    vals.iter().sum::<f64>() + d
                };
                have_scalar = true;
            }
            Prim::Scale(s) => {
                scalar = if have_scalar {
                    scalar * s
                } else {
                    vals.iter().sum::<f64>() * s
                };
                have_scalar = true;
            }
            Prim::CharCodeSum | Prim::ParseFieldsSum | Prim::FoldPositive => {}
        }
    }
    if have_scalar {
        scalar
    } else if vals.is_empty() {
        0.0
    } else {
        vals[0]
    }
}

fn eval_string_pipeline(pipeline: &[Prim], s: &str, k: i64) -> f64 {
    let mut scalar = 0.0;
    for prim in pipeline {
        match prim {
            Prim::CharCodeSum => {
                scalar = s.chars().map(|c| c as u32 as f64).sum();
            }
            Prim::ParseFieldsSum => {
                scalar = s
                    .split(|c: char| !c.is_ascii_digit() && c != '.' && c != '-')
                    .filter(|p| !p.is_empty())
                    .filter_map(|p| p.parse::<f64>().ok())
                    .sum();
            }
            Prim::FoldPositive => {
                scalar = s
                    .chars()
                    .filter_map(|c| c.to_digit(10))
                    .map(|d| d as f64)
                    .sum();
            }
            Prim::Scale(s) => scalar *= s,
            Prim::Offset(d) => scalar += d,
            _ => {}
        }
    }
    let _ = k;
    scalar
}

fn eval_function(f: &FunctionDef, xs: &[f64], k: i64, ys: Option<&[f64]>, s: Option<&str>) -> f64 {
    if xs.is_empty()
        && matches!(
            f.sig,
            SigKind::Xs | SigKind::XsK | SigKind::XsW | SigKind::XsYs
        )
    {
        return 0.0;
    }
    if k < 1 && matches!(f.sig, SigKind::XsK) {
        return if xs.is_empty() {
            0.0
        } else {
            eval_slice_pipeline(&f.pipeline, xs, k, ys)
        };
    }
    match f.sig {
        SigKind::XsK | SigKind::XsW => eval_slice_pipeline(&f.pipeline, xs, k, ys),
        SigKind::Xs => eval_slice_pipeline(&f.pipeline, xs, 0, ys),
        SigKind::XsYs => eval_slice_pipeline(&f.pipeline, xs, k, ys),
        SigKind::SK => {
            if s.is_some_and(|v| v.is_empty()) {
                return 0.0;
            }
            eval_string_pipeline(&f.pipeline, s.unwrap_or("a1b2"), k)
        }
    }
}

fn signature_line(f: &FunctionDef) -> String {
    match f.sig {
        SigKind::XsK => format!("func {}(xs []float64, k int) float64", f.name),
        SigKind::Xs => format!("func {}(xs []float64) float64", f.name),
        SigKind::XsYs => format!("func {}(xs []float64, ys []float64) float64", f.name),
        SigKind::XsW => format!("func {}(xs []float64, w int) float64", f.name),
        SigKind::SK => format!("func {}(s string, k int) float64", f.name),
    }
}

fn describe_pipeline(f: &FunctionDef) -> String {
    let mut parts = Vec::new();
    for prim in &f.pipeline {
        parts.push(match prim {
            Prim::SortAbsDesc => "sort by descending absolute value".into(),
            Prim::TakeFirst(n) => format!("take the first {n} elements"),
            Prim::AlternatingSum => "alternating-sign sum".into(),
            Prim::ScaleInvK(n) => format!("scale by 1/{n}"),
            Prim::Sum => "sum".into(),
            Prim::Product => "product".into(),
            Prim::Mean => "mean".into(),
            Prim::MaxAbs => "maximum absolute value".into(),
            Prim::MinAbs => "minimum absolute value".into(),
            Prim::WindowSum(w) => format!("sum of the first {w} elements"),
            Prim::CountAbove(th) => format!("count of values above {th:.2}"),
            Prim::DotPair => "dot product with a second slice".into(),
            Prim::Clip(lo, hi) => format!("clip each value to [{lo:.1}, {hi:.1}]"),
            Prim::AbsValues => "absolute value of each element".into(),
            Prim::Reverse => "reverse order".into(),
            Prim::Offset(d) => format!("add offset {d:.1}"),
            Prim::Scale(s) => format!("multiply by {s:.2}"),
            Prim::CharCodeSum => "sum of Unicode code points".into(),
            Prim::ParseFieldsSum => "sum of parsed numeric fields".into(),
            Prim::FoldPositive => "sum of decimal digits".into(),
        });
    }
    format!("{} returns the {}.", f.name, parts.join(", then "))
}

fn render_doc(f: &FunctionDef, examples: &[(String, f64)]) -> String {
    let mut doc = String::new();
    doc.push_str(&signature_line(f));
    doc.push('\n');
    doc.push_str(&describe_pipeline(f));
    doc.push_str(" Returns 0 when the input is empty");
    if matches!(f.sig, SigKind::XsK) {
        doc.push_str(" or k < 1");
    }
    doc.push('.');
    doc.push('\n');
    for (idx, (args, want)) in examples.iter().enumerate() {
        doc.push_str(&format!("Example: {}({}) = {:.4}\n", f.name, args, want));
        if idx + 1 >= 2 {
            break;
        }
    }
    doc.trim_end().to_string()
}

fn example_cases(f: &FunctionDef) -> Vec<(String, f64)> {
    let k = f.constants.k as i64;
    let w = f.constants.w as i64;
    type EvalCase<'a> = (&'a [f64], i64, Option<&'a [f64]>, Option<&'a str>);
    let cases: Vec<EvalCase<'_>> = match f.sig {
        SigKind::XsK => vec![
            (&[3.0, -7.0, 2.0][..], k, None, None),
            (&[1.0][..], 3, None, None),
            (&[-2.0, 4.0, -1.0, 5.0][..], k, None, None),
        ],
        SigKind::Xs => vec![
            (&[3.0, -7.0, 2.0][..], 0, None, None),
            (&[1.0][..], 0, None, None),
            (&[] as &[f64], 0, None, None),
        ],
        SigKind::XsYs => vec![
            (&[1.0, 2.0, 3.0][..], 0, Some(&[4.0, 5.0, 6.0][..]), None),
            (&[2.0, -1.0][..], 0, Some(&[3.0, 4.0][..]), None),
            (&[0.5][..], 0, Some(&[2.0][..]), None),
        ],
        SigKind::XsW => vec![
            (&[3.0, -7.0, 2.0, 4.0][..], w, None, None),
            (&[1.0, 2.0, 3.0][..], w, None, None),
            (&[] as &[f64], w, None, None),
        ],
        SigKind::SK => vec![
            (&[][..], k, None, Some("ab9")),
            (&[][..], k, None, Some("x1,y2")),
            (&[][..], 1, None, Some("")),
        ],
    };
    cases
        .into_iter()
        .map(|(xs, kk, ys, s)| {
            let args = match f.sig {
                SigKind::XsK => format!("[{}], {}", fmt_slice(xs), kk),
                SigKind::Xs => format!("[{}]", fmt_slice(xs)),
                SigKind::XsYs => format!("[{}], [{}]", fmt_slice(xs), fmt_slice(ys.unwrap_or(xs))),
                SigKind::XsW => format!("[{}], {}", fmt_slice(xs), kk),
                SigKind::SK => format!("{:?}, {}", s.unwrap_or(""), kk),
            };
            let want = eval_function(f, xs, kk, ys, s);
            (args, want)
        })
        .collect()
}

fn fmt_slice(xs: &[f64]) -> String {
    xs.iter()
        .map(|v| {
            if v.fract() == 0.0 {
                format!("{:.0}", *v as i64)
            } else {
                format!("{v}")
            }
        })
        .collect::<Vec<_>>()
        .join(", ")
}

fn needs_vals_copy(pipeline: &[Prim], sig: SigKind) -> bool {
    if matches!(sig, SigKind::SK) {
        return false;
    }
    pipeline.iter().any(|p| {
        matches!(
            p,
            Prim::SortAbsDesc
                | Prim::TakeFirst(_)
                | Prim::Clip(_, _)
                | Prim::AbsValues
                | Prim::Reverse
        )
    })
}

fn render_go_body(f: &FunctionDef) -> String {
    let mut body = String::new();
    match f.sig {
        SigKind::XsK => {
            body.push_str("    if len(xs) == 0 || k < 1 {\n        return 0\n    }\n");
        }
        SigKind::Xs => {
            body.push_str("    if len(xs) == 0 {\n        return 0\n    }\n");
        }
        SigKind::XsYs => {
            body.push_str("    if len(xs) == 0 || len(ys) == 0 {\n        return 0\n    }\n");
        }
        SigKind::XsW => {
            body.push_str("    if len(xs) == 0 || w < 1 {\n        return 0\n    }\n");
        }
        SigKind::SK => {
            body.push_str("    if s == \"\" {\n        return 0\n    }\n");
        }
    }
    if needs_vals_copy(&f.pipeline, f.sig) {
        body.push_str("    vals := append([]float64(nil), xs...)\n");
    }
    body.push_str("    ");
    body.push_str(&render_go_pipeline(f));
    body.push('\n');
    body
}

fn render_go_pipeline(f: &FunctionDef) -> String {
    let mut lines = Vec::new();
    let vals = if needs_vals_copy(&f.pipeline, f.sig) {
        "vals"
    } else {
        "xs"
    };
    let mut scalar: Option<String> = None;
    for prim in &f.pipeline {
        match prim {
            Prim::SortAbsDesc => {
                lines.push(format!(
                    "sort.Slice({vals}, func(i, j int) bool {{ return math.Abs({vals}[i]) > math.Abs({vals}[j]) }})"
                ));
            }
            Prim::TakeFirst(n) => {
                lines.push(format!("if len({vals}) > {n} {{ {vals} = {vals}[:{n}] }}"));
            }
            Prim::AlternatingSum => {
                scalar = Some(format!("alternatingSum({vals})"));
            }
            Prim::ScaleInvK(n) => {
                let s = scalar.unwrap_or_else(|| format!("alternatingSum({vals})"));
                scalar = Some(format!("({s}) / float64({n}.0)"));
            }
            Prim::Sum => scalar = Some(format!("sumSlice({vals})")),
            Prim::Product => scalar = Some(format!("productSlice({vals})")),
            Prim::Mean => scalar = Some(format!("meanSlice({vals})")),
            Prim::MaxAbs => scalar = Some(format!("maxAbs({vals})")),
            Prim::MinAbs => scalar = Some(format!("minAbs({vals})")),
            Prim::WindowSum(w) => scalar = Some(format!("windowSum({vals}, {w})")),
            Prim::CountAbove(th) => scalar = Some(format!("countAbove({vals}, {th})")),
            Prim::DotPair => scalar = Some("dotPair(xs, ys)".into()),
            Prim::Clip(lo, hi) => {
                lines.push(format!("clipSlice({vals}, {lo}, {hi})"));
            }
            Prim::AbsValues => lines.push(format!("absSlice({vals})")),
            Prim::Reverse => lines.push(format!("reverseSlice({vals})")),
            Prim::Offset(d) => {
                let s = scalar.unwrap_or_else(|| format!("sumSlice({vals})"));
                scalar = Some(format!("({s}) + {d}"));
            }
            Prim::Scale(s) => {
                let base = scalar.unwrap_or_else(|| format!("sumSlice({vals})"));
                scalar = Some(format!("({base}) * {s}"));
            }
            Prim::CharCodeSum => scalar = Some("charCodeSum(s)".into()),
            Prim::ParseFieldsSum => scalar = Some("parseFieldsSum(s)".into()),
            Prim::FoldPositive => scalar = Some("foldPositive(s)".into()),
        }
    }
    if !lines.is_empty() {
        let prefix = lines.join("\n    ");
        if let Some(s) = scalar {
            return format!("{prefix}\n    return {s}");
        }
        return format!("{prefix}\n    return {vals}[0]");
    }
    if let Some(s) = scalar {
        format!("return {s}")
    } else {
        format!("return {vals}[0]")
    }
}

fn go_helpers() -> &'static str {
    r#"
func alternatingSum(xs []float64) float64 {
    sign := 1.0
    total := 0.0
    for _, v := range xs {
        total += sign * v
        sign = -sign
    }
    return total
}

func sumSlice(xs []float64) float64 {
    total := 0.0
    for _, v := range xs {
        total += v
    }
    return total
}

func productSlice(xs []float64) float64 {
    total := 1.0
    for _, v := range xs {
        total *= v
    }
    return total
}

func meanSlice(xs []float64) float64 {
    if len(xs) == 0 {
        return 0
    }
    return sumSlice(xs) / float64(len(xs))
}

func maxAbs(xs []float64) float64 {
    best := 0.0
    for _, v := range xs {
        if a := math.Abs(v); a > best {
            best = a
        }
    }
    return best
}

func minAbs(xs []float64) float64 {
    best := math.Inf(1)
    for _, v := range xs {
        if a := math.Abs(v); a < best {
            best = a
        }
    }
    if math.IsInf(best, 1) {
        return 0
    }
    return best
}

func windowSum(xs []float64, w int) float64 {
    if w < 1 {
        return 0
    }
    if w > len(xs) {
        w = len(xs)
    }
    total := 0.0
    for i := 0; i < w; i++ {
        total += xs[i]
    }
    return total
}

func countAbove(xs []float64, th float64) float64 {
    n := 0
    for _, v := range xs {
        if v > th {
            n++
        }
    }
    return float64(n)
}

func dotPair(xs, ys []float64) float64 {
    n := len(xs)
    if len(ys) < n {
        n = len(ys)
    }
    total := 0.0
    for i := 0; i < n; i++ {
        total += xs[i] * ys[i]
    }
    return total
}

func clipSlice(xs []float64, lo, hi float64) {
    for i, v := range xs {
        if v < lo {
            xs[i] = lo
        } else if v > hi {
            xs[i] = hi
        }
    }
}

func absSlice(xs []float64) {
    for i, v := range xs {
        xs[i] = math.Abs(v)
    }
}

func reverseSlice(xs []float64) {
    for i, j := 0, len(xs)-1; i < j; i, j = i+1, j-1 {
        xs[i], xs[j] = xs[j], xs[i]
    }
}

func charCodeSum(s string) float64 {
    total := 0.0
    for _, r := range s {
        total += float64(r)
    }
    return total
}

func parseFieldsSum(s string) float64 {
    total := 0.0
    field := ""
    flush := func() {
        if field == "" {
            return
        }
        if v, err := strconv.ParseFloat(field, 64); err == nil {
            total += v
        }
        field = ""
    }
    for _, r := range s {
        if (r >= '0' && r <= '9') || r == '.' || r == '-' {
            field += string(r)
        } else {
            flush()
        }
    }
    flush()
    return total
}

func foldPositive(s string) float64 {
    total := 0.0
    for _, r := range s {
        if r >= '0' && r <= '9' {
            total += float64(r - '0')
        }
    }
    return total
}
"#
}

fn render_go_function(f: &FunctionDef) -> String {
    format!(
        "// {}\n{} {{\n{}}}\n",
        describe_pipeline(f),
        signature_line(f),
        render_go_body(f)
    )
}

fn fn_tag(id: usize) -> String {
    format!("[fn:{id:03}]")
}

fn tagged_state(id: usize, text: &str) -> String {
    format!("{} {}", fn_tag(id), text)
}

fn write_tsv_row(state: &str, next: &str) -> String {
    format!(
        "{}\t{}\n",
        escape_pair_field(state),
        escape_pair_field(next)
    )
}

fn write_world_tsv_row(state: &str, next: &str, action: &str) -> String {
    format!(
        "{}\t{}\t{}\n",
        escape_pair_field(state),
        escape_pair_field(next),
        action
    )
}

#[derive(Clone)]
struct TaskSpec {
    instruction: String,
    solve_sig: String,
    arg_names: (String, String),
    explicit: bool,
    dual: Option<usize>,
    eval_only: bool,
}

fn solve_signature(f: &FunctionDef, variant: usize) -> (String, (String, String)) {
    let primary_roots = [
        "xs", "values", "data", "nums", "samples", "points", "series", "input",
    ];
    let secondary_roots = ["k", "n", "limit", "top", "width", "count", "size", "window"];
    let suffixes = ["", "Set", "Batch", "Slice", "Input"];
    let root = variant % primary_roots.len();
    let suffix = suffixes[(variant / primary_roots.len()) % suffixes.len()];
    let a = format!("{}{}", primary_roots[root], suffix);
    let b = format!("{}{}", secondary_roots[root], suffix);
    let sig = match f.sig {
        SigKind::XsK => format!("func Solve({a} []float64, {b} int) float64"),
        SigKind::Xs => format!("func Solve({a} []float64) float64"),
        SigKind::XsYs => format!("func Solve({a} []float64, ys []float64) float64"),
        SigKind::XsW => format!("func Solve({a} []float64, {b} int) float64"),
        SigKind::SK => format!("func Solve(s string, {b} int) float64"),
    };
    (sig, (a, b))
}

fn render_gold(f: &FunctionDef, spec: &TaskSpec, partner: Option<&FunctionDef>) -> String {
    let call = match f.sig {
        SigKind::XsK | SigKind::XsW => format!(
            "veclab.{}({}, {})",
            f.name, spec.arg_names.0, spec.arg_names.1
        ),
        SigKind::Xs => format!("veclab.{}({})", f.name, spec.arg_names.0),
        SigKind::XsYs => format!("veclab.{}({}, ys)", f.name, spec.arg_names.0),
        SigKind::SK => format!("veclab.{}(s, {})", f.name, spec.arg_names.1),
    };
    let body = if let (Some(pid), Some(p)) = (spec.dual, partner) {
        let pcall = match p.sig {
            SigKind::XsK | SigKind::XsW => format!(
                "veclab.{}({}, {})",
                p.name, spec.arg_names.0, spec.arg_names.1
            ),
            SigKind::Xs => format!("veclab.{}({})", p.name, spec.arg_names.0),
            SigKind::XsYs => format!("veclab.{}({}, ys)", p.name, spec.arg_names.0),
            SigKind::SK => format!("veclab.{}(s, {})", p.name, spec.arg_names.1),
        };
        let _ = pid;
        format!("return {call} + {pcall}")
    } else {
        format!("return {call}")
    };
    format!(
        "package solution\n\nimport \"{}\"\n\n{} {{\n    {body}\n}}",
        MODULE_PATH, spec.solve_sig
    )
}

fn task_instruction(
    f: &FunctionDef,
    spec: &TaskSpec,
    doc: &str,
    partner: Option<&FunctionDef>,
) -> String {
    if spec.eval_only {
        return eval_task_instruction(f, spec, partner);
    }
    if spec.explicit {
        let mut s = format!(
            "Using the veclab package, write `{}` that returns veclab.{}",
            spec.solve_sig, f.name
        );
        match f.sig {
            SigKind::XsK | SigKind::XsW => {
                s.push_str(&format!("({}, {})", spec.arg_names.0, spec.arg_names.1));
            }
            SigKind::Xs => {
                s.push_str(&format!("({})", spec.arg_names.0));
            }
            SigKind::XsYs => {
                s.push_str(&format!("({}, ys)", spec.arg_names.0));
            }
            SigKind::SK => {
                s.push_str(&format!("(s, {})", spec.arg_names.1));
            }
        }
        if let Some(p) = partner {
            s.push_str(&format!(" plus veclab.{}", p.name));
        }
        s.push_str(" for the input.");
        return s;
    }
    let behavior = describe_pipeline(f);
    let mut s = format!(
        "Write `{}` that {}",
        spec.solve_sig,
        behavior
            .strip_prefix(&format!("{} ", f.name))
            .unwrap_or(&behavior)
    );
    if let Some(p) = partner {
        s.push_str(&format!(
            " Also combine with {} (call veclab.{}).",
            describe_pipeline(p),
            p.name
        ));
    }
    s.push_str(" Return only Go code.");
    let _ = doc;
    s
}

fn eval_task_instruction(
    f: &FunctionDef,
    spec: &TaskSpec,
    partner: Option<&FunctionDef>,
) -> String {
    if spec.explicit {
        let mut s = format!(
            "Evaluation harness: implement `{}` by delegating to veclab.{}",
            spec.solve_sig, f.name
        );
        match f.sig {
            SigKind::XsK | SigKind::XsW => {
                s.push_str(&format!("({}, {})", spec.arg_names.0, spec.arg_names.1));
            }
            SigKind::Xs => s.push_str(&format!("({})", spec.arg_names.0)),
            SigKind::XsYs => s.push_str(&format!("({}, ys)", spec.arg_names.0)),
            SigKind::SK => s.push_str(&format!("(s, {})", spec.arg_names.1)),
        }
        if let Some(p) = partner {
            s.push_str(&format!(" and veclab.{}", p.name));
        }
        return s;
    }
    format!(
        "Evaluation harness: `{}` must {}",
        spec.solve_sig,
        describe_pipeline(f)
            .strip_prefix(&format!("{} ", f.name))
            .unwrap_or(&describe_pipeline(f))
    )
}

fn knowledge_query(f: &FunctionDef, variant: usize, _doc: &str) -> String {
    let stems = [
        format!("how do I use {} in veclab?", f.name),
        format!("what does {} do?", f.name),
        format!("veclab documentation for {}", f.name),
        format!("explain {}", signature_line(f)),
        format!("query: {}", describe_pipeline(f)),
        format!("show the API contract for veclab.{}", f.name),
        format!("look up the reference entry for {}", f.name),
        format!("describe the inputs and output of {}", f.name),
    ];
    let details = [
        "Include the signature.",
        "Include edge-case behavior.",
        "Include the worked examples.",
        "Give the concise official reference.",
        "State the transformation precisely.",
    ];
    format!(
        "{} {}",
        stems[variant % stems.len()],
        details[(variant / stems.len()) % details.len()]
    )
}

fn encoder_reference_query(f: &FunctionDef, variant: usize) -> String {
    let stems = [
        format!("reference request for veclab.{}", f.name),
        format!("summarize the contract of {}", f.name),
        format!("retrieve documentation for {}", signature_line(f)),
        format!("API notes needed for {}", f.name),
        format!("describe how callers should use {}", f.name),
        format!("state the semantics of veclab.{}", f.name),
        format!("developer reference for {}", f.name),
        format!("explain the return value of {}", f.name),
        format!("document the parameters of {}", f.name),
        format!("give a precise description of {}", describe_pipeline(f)),
        format!("look up {} in the veclab manual", f.name),
        format!("what must a caller know about {}", f.name),
    ];
    let details = [
        "Focus on its signature.",
        "Focus on boundary conditions.",
        "Include representative examples.",
        "Describe inputs, outputs, and ordering.",
        "Return a compact reference entry.",
    ];
    format!(
        "{} {}",
        stems[variant % stems.len()],
        details[(variant / stems.len()) % details.len()]
    )
}

fn ensure_unique_rows(label: &str, text: &str) -> Result<()> {
    let mut seen = HashSet::new();
    for (index, row) in text
        .lines()
        .filter(|line| !line.trim().is_empty())
        .enumerate()
    {
        if !seen.insert(row) {
            bail!(
                "{label} contains a duplicate row at one-based row {}",
                index + 1
            );
        }
    }
    Ok(())
}

fn ensure_disjoint_queries(train: &str, validation: &str) -> Result<()> {
    let train_queries = train
        .lines()
        .filter_map(|row| row.split_once('\t').map(|(query, _)| query))
        .collect::<HashSet<_>>();
    if let Some(overlap) = validation
        .lines()
        .filter_map(|row| row.split_once('\t').map(|(query, _)| query))
        .find(|query| train_queries.contains(query))
    {
        bail!("veclab knowledge train/validation query overlap: {overlap}");
    }
    Ok(())
}

fn generate_tasks_for_fn(
    f: &FunctionDef,
    all: &[FunctionDef],
    doc: &str,
    rng: &mut StdRng,
    eval_only: bool,
) -> Vec<TaskSpec> {
    let count = if eval_only {
        EVAL_TASKS_PER_FUNCTION
    } else {
        TASKS_PER_FUNCTION
    };
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let explicit = if eval_only { i == 0 } else { i % 2 == 0 };
        let dual = if eval_only {
            None
        } else if rng.random::<u8>() % 100 < 30 {
            let same_partition = all
                .iter()
                .filter(|g| {
                    (g.id <= SEEN_FUNCTION_MAX) == (f.id <= SEEN_FUNCTION_MAX) && g.id != f.id
                })
                .map(|g| g.id)
                .collect::<Vec<_>>();
            if same_partition.is_empty() {
                None
            } else {
                Some(same_partition[rng.random_range(0..same_partition.len())])
            }
        } else {
            None
        };
        let (solve_sig, arg_names) = solve_signature(f, i + if eval_only { 10_000 } else { 0 });
        out.push(TaskSpec {
            instruction: String::new(),
            solve_sig,
            arg_names,
            explicit,
            dual,
            eval_only,
        });
    }
    for (i, spec) in out.iter_mut().enumerate() {
        let partner = spec.dual.and_then(|pid| all.iter().find(|g| g.id == pid));
        spec.instruction = task_instruction(f, spec, doc, partner);
        let _ = i;
    }
    out
}

fn test_cases_for_solve(f: &FunctionDef) -> Vec<(String, f64)> {
    example_cases(f).into_iter().take(5).collect()
}

fn fmt_slice_go(xs: &[f64]) -> String {
    if xs.is_empty() {
        return "[]float64{}".into();
    }
    format!("[]float64{{{}}}", fmt_slice(xs))
}

fn fmt_go_want(want: f64) -> String {
    if want < 0.0 {
        format!("({want})")
    } else {
        want.to_string()
    }
}

fn render_module_self_test(functions: &[FunctionDef]) -> Result<String> {
    let mut s = String::from("package veclab\n\nimport (\n\t\"math\"\n\t\"testing\"\n)\n\n");
    for f in functions {
        s.push_str(&format!("func Test_{}(t *testing.T) {{\n", f.name));
        for (args, want) in example_cases(f) {
            let call = match f.sig {
                SigKind::XsK | SigKind::XsW => {
                    if let Some((slice, rest)) = args.split_once("], ") {
                        let xs = parse_slice_from_args(&format!("{slice}]"));
                        format!("{}({}, {})", f.name, fmt_slice_go(&xs), rest)
                    } else {
                        format!("{}({})", f.name, args)
                    }
                }
                SigKind::Xs => format!(
                    "{}({})",
                    f.name,
                    fmt_slice_go(&parse_slice_from_args(&args))
                ),
                SigKind::XsYs => {
                    let (left, right) = args.split_once("], [").unwrap_or((&args, ""));
                    let xs = parse_slice_from_args(left);
                    let ys = parse_slice_from_args(right.trim_end_matches(']'));
                    format!("{}({}, {})", f.name, fmt_slice_go(&xs), fmt_slice_go(&ys))
                }
                SigKind::SK => format!("{}({})", f.name, args),
            };
            s.push_str(&format!(
                "\tif got := {call}; math.Abs(got-{}) > 1e-9 {{ t.Fatalf(\"got %v want {want}\", got) }}\n",
                fmt_go_want(want),
                want = want
            ));
        }
        s.push_str("}\n\n");
    }
    Ok(s)
}

fn render_harness_test(f: &FunctionDef, spec: &TaskSpec) -> String {
    let mut s = String::from("package solution\n\nimport (\n\t\"math\"\n\t\"testing\"\n)\n\n");
    s.push_str("func TestSolve(t *testing.T) {\n");
    for (args, want) in test_cases_for_solve(f) {
        let call = match f.sig {
            SigKind::XsK | SigKind::XsW => {
                if let Some((slice, rest)) = args.split_once("], ") {
                    let xs = parse_slice_from_args(&format!("{slice}]"));
                    format!("Solve({}, {})", fmt_slice_go(&xs), rest)
                } else {
                    format!("Solve({})", args)
                }
            }
            SigKind::Xs => format!("Solve({})", fmt_slice_go(&parse_slice_from_args(&args))),
            SigKind::XsYs => {
                let (left, right) = args.split_once("], [").unwrap_or((&args, ""));
                let xs = parse_slice_from_args(left);
                let ys = parse_slice_from_args(right.trim_end_matches(']'));
                format!("Solve({}, {})", fmt_slice_go(&xs), fmt_slice_go(&ys))
            }
            SigKind::SK => format!("Solve({})", args),
        };
        s.push_str(&format!(
            "\tif got := {call}; math.Abs(got-{}) > 1e-9 {{ t.Fatalf(\"got %v want {want}\", got) }}\n",
            fmt_go_want(want),
            want = want
        ));
    }
    let _ = spec;
    s.push_str("}\n");
    s
}

fn parse_slice_from_args(args: &str) -> Vec<f64> {
    let trimmed = args.trim().trim_start_matches('[').trim_end_matches(']');
    if trimmed.is_empty() {
        return Vec::new();
    }
    trimmed
        .split(',')
        .filter_map(|part| part.trim().parse().ok())
        .collect()
}

#[derive(Serialize)]
struct ManifestFile {
    path: String,
    rows: usize,
    sha256: String,
}

#[derive(Serialize)]
struct Manifest {
    seed: u64,
    generator_version: String,
    files: Vec<ManifestFile>,
}

fn sha256_bytes(data: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data);
    hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn sha256_file(path: &Path) -> Result<String> {
    Ok(sha256_bytes(&fs::read(path)?))
}

pub struct PrepareOptions {
    pub seed: u64,
    pub out: PathBuf,
    pub root: PathBuf,
}

pub fn prepare(opts: PrepareOptions) -> Result<()> {
    let functions = build_functions(opts.seed)?;
    let data_dir = opts.out.clone();
    let module_dir = data_dir.join("veclab");
    let eval_root = opts.root.join("eval/veclab");
    fs::create_dir_all(&module_dir)?;
    fs::create_dir_all(&eval_root)?;

    let mut rng = StdRng::seed_from_u64(opts.seed.wrapping_add(0x9E37_79B9));
    let mut docs_by_id = BTreeMap::new();
    let mut go_impl =
        String::from("package veclab\n\nimport (\n\t\"math\"\n\t\"sort\"\n\t\"strconv\"\n)\n");
    go_impl.push_str(go_helpers());

    for f in &functions {
        let examples = example_cases(f);
        let doc = render_doc(f, &examples);
        docs_by_id.insert(f.id, doc.clone());
        go_impl.push_str(&render_go_function(f));
        go_impl.push('\n');
    }

    fs::write(
        module_dir.join("go.mod"),
        format!("module {MODULE_PATH}\n\ngo 1.22\n"),
    )?;
    fs::write(module_dir.join("veclab.go"), &go_impl)?;
    fs::write(
        module_dir.join("veclab_test.go"),
        render_module_self_test(&functions)?,
    )?;

    let mut docs_rows = String::new();
    for f in &functions {
        let doc = docs_by_id.get(&f.id).unwrap();
        let state = tagged_state(f.id, &signature_line(f));
        docs_rows.push_str(&write_tsv_row(&state, doc));
    }

    let mut knowledge_shards: Vec<Vec<String>> = (0..FUNCTION_COUNT).map(|_| Vec::new()).collect();
    for f in &functions {
        let doc = docs_by_id.get(&f.id).unwrap();
        for v in 0..KNOWLEDGE_ROWS_PER_FUNCTION {
            let query = knowledge_query(f, v, doc);
            let row = write_world_tsv_row(&tagged_state(f.id, &query), doc, "fetch_docs");
            knowledge_shards[f.id - 1].push(row);
        }
    }
    let mut knowledge = String::new();
    let mut knowledge_train = String::new();
    let mut knowledge_val = String::new();
    for round in 0..KNOWLEDGE_ROWS_PER_FUNCTION {
        for shard in &knowledge_shards {
            if let Some(row) = shard.get(round) {
                knowledge.push_str(row);
                if round % 20 == 0 {
                    knowledge_val.push_str(row);
                } else {
                    knowledge_train.push_str(row);
                }
            }
        }
    }

    let mut train = String::new();
    let mut heldout = String::new();
    let mut encoder_mix = String::new();
    let mut eval_lines = Vec::new();
    let mut training_paraphrases = HashSet::new();

    for f in &functions {
        let doc = docs_by_id.get(&f.id).unwrap();
        encoder_mix.push_str(&write_tsv_row(&tagged_state(f.id, &signature_line(f)), doc));

        let train_tasks = generate_tasks_for_fn(f, &functions, doc, &mut rng, false);
        for (ti, spec) in train_tasks.iter().enumerate() {
            let partner = spec
                .dual
                .and_then(|pid| functions.iter().find(|g| g.id == pid));
            let gold = render_gold(f, spec, partner);
            let instr = spec.instruction.clone();
            training_paraphrases.insert(normalize_text(&instr));
            let row = write_tsv_row(&tagged_state(f.id, &instr), &gold);
            encoder_mix.push_str(&write_tsv_row(&tagged_state(f.id, &instr), &instr));
            if f.id <= SEEN_FUNCTION_MAX {
                train.push_str(&row);
            } else {
                heldout.push_str(&row);
            }
            let _ = ti;
        }

        let eval_tasks = generate_tasks_for_fn(f, &functions, doc, &mut rng, true);
        for (ei, spec) in eval_tasks.iter().enumerate() {
            let partner = spec
                .dual
                .and_then(|pid| functions.iter().find(|g| g.id == pid));
            let gold = render_gold(f, spec, partner);
            let instr = spec.instruction.clone();
            if training_paraphrases.contains(&normalize_text(&instr)) {
                bail!(
                    "eval paraphrase collides with training text for fn {}",
                    f.id
                );
            }
            let eval_id = format!("veclab-{}-{}", f.id, ei);
            let harness_dir = eval_root.join(format!("{}-{}", f.id, ei));
            fs::create_dir_all(&harness_dir)?;
            let mut must_call = vec![f.name.clone()];
            if let Some(p) = partner {
                must_call.push(p.name.clone());
            }
            fs::write(
                harness_dir.join("go.mod"),
                format!(
                    "module veclab_eval\n\ngo 1.22\n\nrequire {MODULE_PATH} v0.0.0\nreplace {MODULE_PATH} => ../../../data/fictional/veclab\n"
                ),
            )?;
            fs::write(harness_dir.join("solution.go"), &gold)?;
            fs::write(
                harness_dir.join("main_test.go"),
                render_harness_test(f, spec),
            )?;
            let mut fn_ids = vec![f.id];
            if let Some(p) = partner {
                fn_ids.push(p.id);
            }
            eval_lines.push(serde_json::json!({
                "id": eval_id,
                "fn_ids": fn_ids,
                "subset": if f.id <= SEEN_FUNCTION_MAX { "seen" } else { "heldout" },
                "task": instr,
                "must_call": must_call,
                "harness_dir": format!("eval/veclab/{}-{}/", f.id, ei),
            }));
        }
    }

    // The base rows contribute one signature/document pair plus forty task
    // pairs per function (8,200 rows). Fill the remaining 11,800 rows with
    // 59 distinct reference prompts per function. Repeating a single template
    // here previously made 80% of nominal encoder examples duplicates.
    for variant in 0..59 {
        for f in &functions {
            let doc = docs_by_id.get(&f.id).unwrap();
            encoder_mix.push_str(&write_tsv_row(
                &tagged_state(f.id, &encoder_reference_query(f, variant)),
                doc,
            ));
        }
    }
    if encoder_mix.lines().count() != 20_000 {
        bail!(
            "veclab encoder corpus must contain exactly 20000 rows, got {}",
            encoder_mix.lines().count()
        );
    }
    ensure_unique_rows("veclab knowledge train", &knowledge_train)?;
    ensure_unique_rows("veclab knowledge validation", &knowledge_val)?;
    ensure_disjoint_queries(&knowledge_train, &knowledge_val)?;
    ensure_unique_rows("veclab task train", &train)?;
    ensure_unique_rows("veclab task heldout", &heldout)?;
    ensure_unique_rows("veclab encoder", &encoder_mix)?;

    fs::write(data_dir.join("veclab_docs.txt"), &docs_rows)?;
    fs::write(data_dir.join("veclab_knowledge.txt"), &knowledge)?;
    fs::write(
        data_dir.join("veclab_knowledge_train.txt"),
        &knowledge_train,
    )?;
    fs::write(data_dir.join("veclab_knowledge_val.txt"), &knowledge_val)?;
    fs::write(data_dir.join("veclab_tasks_train.txt"), &train)?;
    fs::write(data_dir.join("veclab_tasks_heldout.txt"), &heldout)?;
    fs::write(data_dir.join("veclab_encoder_mix.txt"), &encoder_mix)?;
    fs::write(
        opts.root.join("eval/veclab_eval.jsonl"),
        eval_lines
            .iter()
            .map(|v| serde_json::to_string(v).unwrap())
            .collect::<Vec<_>>()
            .join("\n")
            + "\n",
    )?;

    verify_go_module(&module_dir)?;
    verify_harnesses(&eval_root, &functions)?;
    run_leak_guards(
        &data_dir,
        &eval_root,
        &training_paraphrases,
        &eval_lines
            .iter()
            .filter_map(|v| v.get("task").and_then(|t| t.as_str()))
            .collect::<Vec<_>>(),
    )?;

    let files = [
        ("veclab_docs.txt", docs_rows.lines().count()),
        ("veclab_knowledge.txt", knowledge.lines().count()),
        (
            "veclab_knowledge_train.txt",
            knowledge_train.lines().count(),
        ),
        ("veclab_knowledge_val.txt", knowledge_val.lines().count()),
        ("veclab_tasks_train.txt", train.lines().count()),
        ("veclab_tasks_heldout.txt", heldout.lines().count()),
        ("veclab_encoder_mix.txt", encoder_mix.lines().count()),
    ];
    let mut manifest_files = Vec::new();
    for (name, rows) in files {
        let path = data_dir.join(name);
        manifest_files.push(ManifestFile {
            path: format!("data/fictional/{name}"),
            rows,
            sha256: sha256_file(&path)?,
        });
    }
    let eval_path = opts.root.join("eval/veclab_eval.jsonl");
    manifest_files.push(ManifestFile {
        path: "eval/veclab_eval.jsonl".into(),
        rows: eval_lines.len(),
        sha256: sha256_file(&eval_path)?,
    });
    let manifest = Manifest {
        seed: opts.seed,
        generator_version: GENERATOR_VERSION.into(),
        files: manifest_files,
    };
    fs::write(
        data_dir.join("MANIFEST.json"),
        serde_json::to_string_pretty(&manifest)?,
    )?;

    println!(
        "Prepared veclab corpus: {} functions, {} eval cases, seed {}",
        FUNCTION_COUNT,
        eval_lines.len(),
        opts.seed
    );
    Ok(())
}

fn normalize_text(s: &str) -> String {
    s.to_ascii_lowercase()
        .chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .collect()
}

fn verify_go_module(module_dir: &Path) -> Result<()> {
    for cmd in [["vet", "./..."], ["test", "./..."]] {
        let status = Command::new("go")
            .args(cmd)
            .current_dir(module_dir)
            .status()
            .context("go toolchain not available")?;
        if !status.success() {
            bail!("go {} failed in {}", cmd[0], module_dir.display());
        }
    }
    Ok(())
}

fn verify_harnesses(eval_root: &Path, functions: &[FunctionDef]) -> Result<()> {
    functions.par_iter().try_for_each(|f| -> Result<()> {
        for ei in 0..EVAL_TASKS_PER_FUNCTION {
            let dir = eval_root.join(format!("{}-{}", f.id, ei));
            let gold = fs::read_to_string(dir.join("solution.go"))?;
            let gold_ok = Command::new("go")
                .args(["test", "."])
                .current_dir(&dir)
                .env("GOMAXPROCS", "1")
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status()?
                .success();
            if !gold_ok {
                bail!("gold solution failed harness {}", dir.display());
            }
            let stub = stub_solve_source(f);
            fs::write(dir.join("solution.go"), &stub)?;
            let stub_ok = Command::new("go")
                .args(["test", "."])
                .current_dir(&dir)
                .env("GOMAXPROCS", "1")
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status()?
                .success();
            if stub_ok {
                bail!("stub incorrectly passed harness {}", dir.display());
            }
            fs::write(dir.join("solution.go"), gold)?;
        }
        Ok(())
    })
}

fn stub_solve_source(f: &FunctionDef) -> String {
    match f.sig {
        SigKind::XsK => {
            "package solution\n\nfunc Solve(xs []float64, k int) float64 { return 0 }\n"
        }
        SigKind::Xs => "package solution\n\nfunc Solve(xs []float64) float64 { return 0 }\n",
        SigKind::XsYs => {
            "package solution\n\nfunc Solve(xs []float64, ys []float64) float64 { return 0 }\n"
        }
        SigKind::XsW => {
            "package solution\n\nfunc Solve(xs []float64, w int) float64 { return 0 }\n"
        }
        SigKind::SK => "package solution\n\nfunc Solve(s string, k int) float64 { return 0 }\n",
    }
    .to_string()
}

fn run_leak_guards(
    data_dir: &Path,
    eval_root: &Path,
    training_paraphrases: &HashSet<String>,
    eval_tasks: &[&str],
) -> Result<()> {
    let train_text = fs::read_to_string(data_dir.join("veclab_tasks_train.txt"))?;
    let encoder_text = fs::read_to_string(data_dir.join("veclab_encoder_mix.txt"))?;
    let knowledge_text = fs::read_to_string(data_dir.join("veclab_knowledge.txt"))?;
    if let Some(row) = heldout_gold_in(&train_text).into_iter().next() {
        bail!("held-out gold leaked into train: fn {}", row);
    }
    if let Some(row) = heldout_gold_in(&encoder_text).into_iter().next() {
        bail!("held-out gold leaked into encoder mix: fn {}", row);
    }
    if let Some(row) = heldout_gold_in(&knowledge_text).into_iter().next() {
        bail!("held-out gold leaked into knowledge: fn {}", row);
    }
    if train_text.contains("data/fictional/veclab") {
        bail!("implementation path leaked into train data");
    }
    for eval_task in eval_tasks {
        let norm = normalize_text(eval_task);
        if training_paraphrases.contains(&norm) {
            bail!("eval paraphrase appears in training data");
        }
    }
    let knowledge = fs::read_to_string(data_dir.join("veclab_knowledge.txt"))?;
    for window in knowledge.lines().collect::<Vec<_>>().windows(WORLD_BATCH) {
        let mut seen = HashSet::new();
        for line in window {
            let state = line.split('\t').next().unwrap_or(line);
            if let Some(id) = parse_fn_tag(state) {
                if !seen.insert(id) {
                    bail!("knowledge batch window has duplicate fn id {id}");
                }
            }
        }
    }
    let impl_path = data_dir.join("veclab");
    if !impl_path.is_dir() {
        bail!("missing hidden veclab module at {}", impl_path.display());
    }
    let _ = eval_root;
    Ok(())
}

fn parse_fn_tag(line: &str) -> Option<usize> {
    let start = line.find("[fn:")? + 4;
    let rest = &line[start..];
    let end = rest.find(']')?;
    rest[..end].parse().ok()
}

fn heldout_gold_in(text: &str) -> Vec<usize> {
    text.lines()
        .filter_map(|line| {
            let mut fields = line.split('\t');
            let state = fields.next()?;
            let next = fields.next().unwrap_or("");
            let id = parse_fn_tag(state)?;
            if id > SEEN_FUNCTION_MAX && next.contains("package solution") {
                Some(id)
            } else {
                None
            }
        })
        .collect()
}

pub fn print_split_stats(data_dir: &Path) -> Result<()> {
    let train = fs::read_to_string(data_dir.join("veclab_tasks_train.txt"))?;
    let heldout = fs::read_to_string(data_dir.join("veclab_tasks_heldout.txt"))?;
    let docs = fs::read_to_string(data_dir.join("veclab_docs.txt"))?;
    let encoder = fs::read_to_string(data_dir.join("veclab_encoder_mix.txt"))?;
    let knowledge = fs::read_to_string(data_dir.join("veclab_knowledge.txt"))?;
    let knowledge_train = fs::read_to_string(data_dir.join("veclab_knowledge_train.txt"))?;
    let knowledge_val = fs::read_to_string(data_dir.join("veclab_knowledge_val.txt"))?;
    let train_leak = heldout_gold_in(&train).len();
    let encoder_leak = heldout_gold_in(&encoder).len();
    let knowledge_leak = heldout_gold_in(&knowledge).len();
    println!(
        "veclab split stats: docs_rows={} knowledge_rows={} knowledge_train_rows={} knowledge_val_rows={} task_train_rows={} task_heldout_rows={} encoder_rows={} heldout_gold_upstream={}",
        docs.lines().filter(|line| !line.trim().is_empty()).count(),
        knowledge.lines().filter(|line| !line.trim().is_empty()).count(),
        knowledge_train.lines().filter(|line| !line.trim().is_empty()).count(),
        knowledge_val.lines().filter(|line| !line.trim().is_empty()).count(),
        train.lines().count(),
        heldout.lines().count(),
        encoder.lines().filter(|line| !line.trim().is_empty()).count(),
        train_leak + encoder_leak + knowledge_leak
    );
    if train_leak + encoder_leak + knowledge_leak != 0 {
        bail!("held-out gold rows leaked upstream");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_under_fixed_seed() -> Result<()> {
        let a = build_functions(42)?;
        let b = build_functions(42)?;
        assert_eq!(a.len(), b.len());
        assert_eq!(a[0].name, b[0].name);
        assert_eq!(a[0].pipeline.len(), b[0].pipeline.len());
        Ok(())
    }

    #[test]
    fn generated_training_variants_are_unique() -> Result<()> {
        let functions = build_functions(42)?;
        let function = &functions[0];
        let knowledge = (0..KNOWLEDGE_ROWS_PER_FUNCTION)
            .map(|variant| knowledge_query(function, variant, ""))
            .collect::<HashSet<_>>();
        let signatures = (0..TASKS_PER_FUNCTION)
            .map(|variant| solve_signature(function, variant).0)
            .collect::<HashSet<_>>();
        let encoder_references = (0..59)
            .map(|variant| encoder_reference_query(function, variant))
            .collect::<HashSet<_>>();
        assert_eq!(knowledge.len(), KNOWLEDGE_ROWS_PER_FUNCTION);
        assert_eq!(signatures.len(), TASKS_PER_FUNCTION);
        assert_eq!(encoder_references.len(), 59);
        Ok(())
    }

    #[test]
    fn knowledge_round_robin_has_unique_batch_window() -> Result<()> {
        let mut shards: Vec<Vec<usize>> = (0..FUNCTION_COUNT).map(|_| Vec::new()).collect();
        for id in 1..=FUNCTION_COUNT {
            for round in 0..KNOWLEDGE_ROWS_PER_FUNCTION {
                shards[id - 1].push(id);
                let _ = round;
            }
        }
        let mut knowledge = Vec::new();
        for round in 0..KNOWLEDGE_ROWS_PER_FUNCTION {
            for shard in &shards {
                if let Some(id) = shard.get(round) {
                    knowledge.push(*id);
                }
            }
        }
        for window in knowledge.windows(WORLD_BATCH) {
            let mut seen = HashSet::new();
            for id in window {
                assert!(seen.insert(*id), "duplicate fn id {id} in batch window");
            }
        }
        Ok(())
    }

    #[test]
    fn knowledge_split_holds_out_paraphrases_for_every_function() {
        let mut train_ids = HashSet::new();
        let mut val_ids = HashSet::new();
        for round in 0..KNOWLEDGE_ROWS_PER_FUNCTION {
            for function_id in 1..=FUNCTION_COUNT {
                if round % 20 == 0 {
                    val_ids.insert(function_id);
                } else {
                    train_ids.insert(function_id);
                }
            }
        }
        assert_eq!(train_ids.len(), FUNCTION_COUNT);
        assert_eq!(val_ids.len(), FUNCTION_COUNT);
    }

    #[test]
    fn knowledge_rows_use_the_fetch_docs_action() {
        let row = write_world_tsv_row("[fn:001] query", "documentation", "fetch_docs");
        let parsed = crate::data::data::raw_world_example_from_line_with_mode(
            row.trim(),
            crate::data::TokenizationMode::Default,
        )
        .expect("world row should parse");
        assert_eq!(parsed.action_label, crate::data::ACTION_FETCH_DOCS);
    }
}
