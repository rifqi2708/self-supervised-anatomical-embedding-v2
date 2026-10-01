# Shared evidence contract for reviewed-mask matching

Status: implemented engineering decision; scientific pilot gates remain pending.
Source: approved GitHub spec #3 and work items 4–12 (#7–#15).

UAE and registration execute independently but must preserve the same frozen
query identities, crop/context provenance, coordinate conventions and input
contract. Both therefore export one common method manifest, outcome CSV and
checksummed member inventory. Existing cohort commands expose these adapters;
legacy cohort commands and their output contracts remain available.

Result analysis validates each bundle against the frozen contract before joining
methods. Pairwise shared-valid queries define conditional median/p95 contrasts,
while failure rates retain all attempted queries. Anatomical review has a
separate versioned ledger and discovery/confirmation gate.

Checkpoints use external diagnostic staging and sibling snapshot promotion.
This preserves the last verified bundle when a new artifact conflicts or a write
fails. Hard links avoid duplicating immutable retained artifacts. Previous
snapshots remain preserved, with storage and hashing overhead subject to pilot
measurement. A full evidence checksum does not replace independent dataset,
checkpoint, cache or Git recovery verification.
