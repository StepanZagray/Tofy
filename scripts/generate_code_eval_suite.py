#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def task(id_, prompt, harness, must_contain=None, must_not_contain=None, tags=None, max_new_tokens=384):
    return {
        "id": id_,
        "language": "rust",
        "expected_action": "code",
        "prompt": prompt,
        "harness_template": harness,
        "must_contain": must_contain or [],
        "must_not_contain": must_not_contain or ["todo!(", "unimplemented!(", "panic!("],
        "tags": tags or ["rust", "code-first", "unit-test"],
        "max_new_tokens": max_new_tokens,
    }


TASKS = [
    task(
        "normalize_path",
        """Return only Rust code. Implement exactly this function:
pub fn normalize_path(input: &str) -> String

Rules:
- Unix-style paths only.
- Collapse repeated slashes.
- Remove '.' segments.
- Resolve '..' without going above root for absolute paths.
- Preserve leading '..' segments for relative paths.
- Return '/' for empty absolute result and '.' for empty relative result.
""",
        """{{code}}

#[cfg(test)]
mod tests {
    use super::normalize_path;

    #[test]
    fn normalizes_absolute_paths() {
        assert_eq!(normalize_path("/a//b/./c/../d"), "/a/b/d");
        assert_eq!(normalize_path("/../../a"), "/a");
        assert_eq!(normalize_path("///"), "/");
    }

    #[test]
    fn normalizes_relative_paths() {
        assert_eq!(normalize_path("./a/./b"), "a/b");
        assert_eq!(normalize_path("a/b/../.."), ".");
        assert_eq!(normalize_path("../../a"), "../../a");
    }
}
""",
        must_contain=["pub fn normalize_path"],
    ),
    task(
        "parse_size",
        """Return only Rust code. Implement exactly this function:
pub fn parse_size(input: &str) -> Result<u64, String>

Rules:
- Support B, KB, MB, and GB units with base 1024.
- Unit matching is ASCII case-insensitive.
- Allow surrounding whitespace and optional whitespace between number and unit.
- Reject empty input, decimals, negatives, and unknown units.
""",
        """{{code}}

#[cfg(test)]
mod tests {
    use super::parse_size;

    #[test]
    fn parses_valid_sizes() {
        assert_eq!(parse_size("42").unwrap(), 42);
        assert_eq!(parse_size("2 KB").unwrap(), 2 * 1024);
        assert_eq!(parse_size("3mb").unwrap(), 3 * 1024 * 1024);
        assert_eq!(parse_size("1Gb").unwrap(), 1024 * 1024 * 1024);
    }

    #[test]
    fn rejects_invalid_sizes() {
        assert!(parse_size("").is_err());
        assert!(parse_size("-1KB").is_err());
        assert!(parse_size("1.5MB").is_err());
        assert!(parse_size("4TB").is_err());
    }
}
""",
        must_contain=["pub fn parse_size"],
    ),
    task(
        "merge_intervals",
        """Return only Rust code. Implement exactly this function:
pub fn merge_intervals(intervals: &[(i32, i32)]) -> Vec<(i32, i32)>

Rules:
- Normalize reversed intervals like (5, 3) into (3, 5).
- Sort by start ascending, then end ascending.
- Merge overlapping intervals and also merge touching intervals where next.start <= current.end + 1.
""",
        """{{code}}

#[cfg(test)]
mod tests {
    use super::merge_intervals;

    #[test]
    fn merges_and_normalizes() {
        assert_eq!(
            merge_intervals(&[(5, 3), (4, 8), (20, 21), (22, 22)]),
            vec![(3, 8), (20, 22)]
        );
    }

    #[test]
    fn handles_empty_and_separate() {
        assert_eq!(merge_intervals(&[]), Vec::<(i32, i32)>::new());
        assert_eq!(merge_intervals(&[(1, 1), (3, 3)]), vec![(1, 1), (3, 3)]);
    }
}
""",
        must_contain=["pub fn merge_intervals"],
    ),
    task(
        "dedup_case_insensitive",
        """Return only Rust code. Implement exactly this function:
pub fn dedup_case_insensitive(values: &[&str]) -> Vec<String>

Rules:
- Compare using ASCII lowercase after trimming surrounding whitespace.
- Skip entries that become empty after trimming.
- Preserve the first original trimmed spelling for each unique lowercase key.
""",
        """{{code}}

#[cfg(test)]
mod tests {
    use super::dedup_case_insensitive;

    #[test]
    fn preserves_first_spelling() {
        assert_eq!(
            dedup_case_insensitive(&["  Foo", "foo", "BAR", "bar ", "Baz"]),
            vec!["Foo".to_string(), "BAR".to_string(), "Baz".to_string()]
        );
    }

    #[test]
    fn skips_empty_values() {
        assert_eq!(
            dedup_case_insensitive(&["", "   ", " Quux "]),
            vec!["Quux".to_string()]
        );
    }
}
""",
        must_contain=["pub fn dedup_case_insensitive"],
    ),
    task(
        "top_k_words",
        """Return only Rust code. Implement exactly this function:
pub fn top_k_words(words: &[&str], k: usize) -> Vec<(String, usize)>

Rules:
- Trim each word and ignore empties.
- Compare case-insensitively using ASCII lowercase.
- Sort by frequency descending, then word ascending.
- Return at most k entries.
""",
        """{{code}}

#[cfg(test)]
mod tests {
    use super::top_k_words;

    #[test]
    fn ranks_words() {
        assert_eq!(
            top_k_words(&[" Rust ", "go", "rust", "Go", "zig", "rust"], 2),
            vec![("rust".to_string(), 3), ("go".to_string(), 2)]
        );
    }

    #[test]
    fn handles_zero_and_ties() {
        assert_eq!(top_k_words(&["b", "a", "B", "a"], 0), vec![]);
        assert_eq!(
            top_k_words(&["b", "a", "B", "a"], 3),
            vec![("a".to_string(), 2), ("b".to_string(), 2)]
        );
    }
}
""",
        must_contain=["pub fn top_k_words"],
    ),
    task(
        "longest_balanced_prefix",
        """Return only Rust code. Implement exactly this function:
pub fn longest_balanced_prefix(input: &str) -> usize

Rules:
- Track only (), [], and {}.
- Ignore all other characters.
- Return the byte index immediately after the longest prefix that is valid and fully balanced.
- If the prefix becomes invalid because of a mismatched closing delimiter, stop there.
""",
        """{{code}}

#[cfg(test)]
mod tests {
    use super::longest_balanced_prefix;

    #[test]
    fn finds_balanced_prefixes() {
        assert_eq!(longest_balanced_prefix("([])x"), 4);
        assert_eq!(longest_balanced_prefix("{a[b]c}tail"), 7);
        assert_eq!(longest_balanced_prefix("abc"), 3);
    }

    #[test]
    fn stops_on_invalid_prefixes() {
        assert_eq!(longest_balanced_prefix("([)]tail"), 0);
        assert_eq!(longest_balanced_prefix("ok()]}"), 4);
    }
}
""",
        must_contain=["pub fn longest_balanced_prefix"],
    ),
    task(
        "parse_header_block",
        """Return only Rust code. Implement exactly this function:
pub fn parse_header_block(input: &str) -> Result<Vec<(String, String)>, String>

Rules:
- Each non-empty line must be 'Key: Value'.
- Trim outer whitespace around key and value.
- Lowercase keys using ASCII lowercase.
- Reject empty keys and duplicate keys.
""",
        """{{code}}

#[cfg(test)]
mod tests {
    use super::parse_header_block;

    #[test]
    fn parses_headers() {
        assert_eq!(
            parse_header_block("Host: example.com\\n X-Mode : Fast ").unwrap(),
            vec![
                ("host".to_string(), "example.com".to_string()),
                ("x-mode".to_string(), "Fast".to_string())
            ]
        );
    }

    #[test]
    fn rejects_bad_headers() {
        assert!(parse_header_block("NoColon").is_err());
        assert!(parse_header_block(": value").is_err());
        assert!(parse_header_block("A: x\\na: y").is_err());
    }
}
""",
        must_contain=["pub fn parse_header_block"],
    ),
    task(
        "compact_sorted_numbers",
        """Return only Rust code. Implement exactly this function:
pub fn compact_sorted_numbers(nums: &[i32]) -> Vec<String>

Rules:
- Input is sorted ascending but may contain duplicates.
- Collapse duplicates before formatting.
- Runs of length 1 become 'n'.
- Runs of length 2 become 'a,b'.
- Runs of length >= 3 become 'a-b'.
""",
        """{{code}}

#[cfg(test)]
mod tests {
    use super::compact_sorted_numbers;

    #[test]
    fn compacts_ranges() {
        assert_eq!(
            compact_sorted_numbers(&[1, 2, 3, 5, 6, 9, 9, 10, 11, 12]),
            vec!["1-3".to_string(), "5,6".to_string(), "9-12".to_string()]
        );
    }

    #[test]
    fn handles_short_inputs() {
        assert_eq!(compact_sorted_numbers(&[]), Vec::<String>::new());
        assert_eq!(compact_sorted_numbers(&[4]), vec!["4".to_string()]);
    }
}
""",
        must_contain=["pub fn compact_sorted_numbers"],
    ),
    task(
        "retry_schedule",
        """Return only Rust code. Implement exactly this function:
pub fn retry_schedule(base_ms: u64, factor: u64, attempts: usize, max_ms: u64) -> Vec<u64>

Rules:
- Return exactly attempts entries.
- First delay is base_ms.
- Each next delay is previous * factor using saturating arithmetic.
- Clamp every returned delay to max_ms.
""",
        """{{code}}

#[cfg(test)]
mod tests {
    use super::retry_schedule;

    #[test]
    fn builds_schedule() {
        assert_eq!(retry_schedule(100, 2, 5, 1_000), vec![100, 200, 400, 800, 1_000]);
    }

    #[test]
    fn handles_edge_cases() {
        assert_eq!(retry_schedule(5, 3, 0, 99), Vec::<u64>::new());
        assert_eq!(retry_schedule(u64::MAX, 4, 2, 50), vec![50, 50]);
    }
}
""",
        must_contain=["pub fn retry_schedule"],
    ),
    task(
        "strip_line_comments",
        """Return only Rust code. Implement exactly this function:
pub fn strip_line_comments(input: &str) -> String

Rules:
- Remove // comments until end of line.
- Ignore // inside double-quoted strings.
- Respect backslash escapes inside strings.
- Preserve line breaks.
""",
        """{{code}}

#[cfg(test)]
mod tests {
    use super::strip_line_comments;

    #[test]
    fn strips_comments() {
        let src = "let a = 1; // comment\\nlet b = 2;//x\\n";
        assert_eq!(strip_line_comments(src), "let a = 1; \\nlet b = 2;\\n");
    }

    #[test]
    fn keeps_comment_markers_inside_strings() {
        let src = "let s = \\"http://example.com\\"; // keep url\\nlet t = \\"a\\\\\\"//b\\"; // tail";
        assert_eq!(
            strip_line_comments(src),
            "let s = \\"http://example.com\\"; \\nlet t = \\"a\\\\\\"//b\\"; "
        );
    }
}
""",
        must_contain=["pub fn strip_line_comments"],
    ),
]


def main():
    parser = argparse.ArgumentParser(description="Generate the tiny hard code-first Rust eval suite.")
    parser.add_argument("--output", default="eval/code_assistant_rust_hard.jsonl")
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as fh:
        for item in TASKS:
            fh.write(json.dumps(item, ensure_ascii=True) + "\n")
    print(f"Wrote {len(TASKS)} tasks to {output}")


if __name__ == "__main__":
    main()
