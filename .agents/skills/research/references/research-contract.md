# Research contract

## Quality gates

A run is complete only when:

1. The brief has a concrete question and completion criterion.
2. Existing library knowledge was searched first.
3. Every material factual claim has traceable evidence.
4. Primary sources were used when reasonably available.
5. Important claims received an independent verification pass.
6. Conflicts, uncertainty, source limitations, and negative results remain visible.
7. Canonical knowledge contains conclusions and citations; raw worker material stays in the run directory.
8. Completed experiment analyses preserve a concise, linked insight note under the
   scope's `insights/` directory, including negative results and the next falsifiable test.

Never use the number of sources or agents as a quality proxy. One directly applicable specification can outweigh ten derivative articles.

## Library layout

```text
<library>/
├── INDEX.md
├── <category>/
│   ├── INDEX.md
│   └── <scope>/                 # optional project or enduring subject
│       ├── INDEX.md
│       ├── <topic>.md
│       └── insights/
│           ├── INDEX.md
│           └── <experiment>.md
└── _runs/
    └── <timestamp>-<topic>/
        ├── brief.md
        ├── manifest.json
        ├── findings/
        │   └── <assignment>.md
        └── synthesis.md
```

Use lowercase hyphenated paths. Categories are broad and stable. Scopes keep project-specific knowledge together without turning every project into a new top-level category.

## Category rules

- Reuse an existing category whenever its meaning fits.
- Use `ml/tofy` for Tofy model architecture, training, evaluation, ARC-AGI, and related machine-learning research.
- Put general machine-learning knowledge directly under `ml` or under another stable scope when one exists.
- Create a category only when the subject is enduring and materially distinct from every existing category.
- Record category descriptions in the relevant `INDEX.md` so future agents classify consistently.

## Finding format

Each worker finding file must contain:

```markdown
# Assignment title

## Answer

Direct answer to the bounded assignment.

## Findings

### Finding title

- Status: established | supported | tentative | disputed | obsolete
- Confidence: high | medium | low
- Claim: One precise claim.
- Evidence: What the source directly establishes.
- Source: [Descriptive title](URL) or an absolute/local repository path with lines.
- Source date: YYYY-MM-DD or unknown
- Retrieved: YYYY-MM-DD for web sources
- Limits: Scope gaps, assumptions, or reasons confidence is not higher.

## Contradictions and gaps

Evidence that disagrees, unavailable primary sources, or unanswered parts.
```

The claim must not be broader than the evidence. Label agent reasoning as inference.

## Verification format

For every high-impact candidate claim, record one verdict:

- `verified`: directly supported and scoped correctly;
- `qualified`: directionally supported but wording or scope must narrow;
- `disputed`: credible evidence conflicts;
- `unsupported`: evidence does not establish the claim;
- `stale`: the claim may once have been correct but its source is no longer current enough.

Include the reason and strongest applicable source. A verifier should try to falsify, not merely summarize.

## Canonical topic note

Use this structure:

```markdown
---
title: "Topic title"
category: "category"
scope: "optional-scope"
status: "current"
created: "YYYY-MM-DD"
updated: "YYYY-MM-DD"
tags: ["tag"]
research_runs: ["relative/run/path"]
---

# Topic title

## Current understanding

The shortest accurate synthesis.

## Established findings

Source-backed findings with citations adjacent to the claims.

## Interpretation

Clearly labeled inferences and decision implications.

## Disputed or uncertain

Contradictions, weak evidence, and unknowns.

## Sources

Deduplicated source list with access dates where relevant.
```

Update a topic note in place as knowledge develops. Preserve previous run references. Mark an obsolete conclusion instead of silently erasing why it changed.

## Run manifest

Use JSON with this minimum shape:

```json
{
  "id": "2026-08-08T120000Z-topic",
  "question": "Research question",
  "category": "ml",
  "scope": "tofy",
  "status": "planning",
  "created_at": "2026-08-08T12:00:00Z",
  "updated_at": "2026-08-08T12:00:00Z",
  "assignments": [],
  "verification": "pending",
  "published_to": null
}
```

Valid run states are `planning`, `researching`, `verifying`, `synthesizing`, `complete`, and `failed`. Each assignment records its own `pending`, `in_progress`, `complete`, `failed`, or `stale` status. Write changes atomically when possible.

## Source discipline

- Browse when information is current, niche, uncertain, high-stakes, or explicitly requested.
- Cite the source that owns the claim, not a search-results page.
- Record source dates and distinguish event dates from publication dates.
- Use local code and repository documents as primary evidence for project-specific behavior.
- Do not cite an agent summary as evidence.
- Do not quote beyond applicable copyright limits; prefer concise paraphrase.
- If the ideal source cannot be accessed, say so and reduce confidence.

## Stopping rule

Stop when the completion criterion is met, material claims pass verification, and further searching is unlikely to change the answer. Stop earlier and report the gap when the needed evidence is unavailable. Never loop merely to consume a nominal research budget.
