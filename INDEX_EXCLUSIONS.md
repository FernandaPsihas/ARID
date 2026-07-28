# Tier 1 indexing exclusions

What `extract.py` drops when indexing the Tier 1 repo set, and why. Written
2026-07-27 while scoping the full-DUNE index (up from the dunereco-only
subset, ~7004 chunks).

Two mechanisms drop files:

1. **The extension allowlist** (`DISPATCH` in `extract.py`) — anything without
   a registered parser is skipped. This is where the binary assets go.
2. **`EXCLUDE_PATTERNS`** (`extract.py`) — path regexes applied after the
   allowlist, for files that parse fine but shouldn't be retrievable.

## Headline numbers

| | Files |
|---|---|
| Tracked files across 24 repos | 9,222 |
| Indexable after extension allowlist | 6,989 |
| Dropped by `EXCLUDE_PATTERNS` | 1,663 |
| **Actually indexed** | **5,326** |

Verified by running `extract.py` over all 24 repos: **43,142 chunks**, against
7,004 for the dunereco-only collection today — a 6.2× scale-up. Per-rule
exclusion totals from that run match the table below exactly.

Note the near-wash against the previous behaviour: the old dispatch indexed
5,338 files. Roughly the same count, almost entirely different composition —
1,651 C++ headers came *in*, 1,663 data tables and job permutations went *out*.

---

## 1. Dropped by extension (no parser registered)

These never reach a parser. No action needed, listed so it's on the record
that the omission is deliberate rather than an oversight.

| Extension | Count | What it is | Why dropped |
|---|---|---|---|
| `.txt` | 713 | `CMakeLists.txt`, plus a 13MB `FDHDChannelMap_WIBEth_visiblewires_v1.txt` channel map in dunecore | Build recipes carry some linkage info, but not enough to justify a parser. The channel map is a numeric table. |
| `.xml` | 601 | duneutil grid job submission configs | Batch submission boilerplate. The package-linkage information duneutil is in Tier 1 *for* lives in `ups/product_deps` and CMake, not here. |
| `.gdml` | 276 | dunecore detector geometry (`iceberg_v3_refactored.gdml` alone is 17MB) | Machine-generated geometry XML. Enormous, no natural language. |
| `.pl` `.sh` | 213 | Build/setup scripts, mostly in larsoft and duneutil | Marginal value, high boilerplate. |
| `.bz2` `.ts` `.root` `.dat` `.vec` `.db` `.png` | ~80 | dunereco WireCell response tensors and noise models (20–35MB each) | Binary. |
| `.md` `.dox` `.tex` | 54 | READMEs and Doxygen pages | Worth revisiting — this is real prose about the codebase. Deferred, not rejected. |
| `.xsd` `.json` `.yml` `.cmake` `.in` `.ipynb` `.mac` | ~120 | Schemas, CI config, misc | Low value. |

**Deliberately kept in the allowlist:** `.h` (1,636), `.tcc` (12), `.c++` (3).
These were being dropped by an earlier bug — see the note at the bottom.

## 2. Dropped by `EXCLUDE_PATTERNS`

These parse correctly and would produce valid chunks. They're excluded because
indexing them makes retrieval *worse*, not just bigger.

### `fcl/protodune/fcldirs/(calib|pedestal|rundata)/` — 456 files

Numeric constant tables wearing a `.fcl` extension.

```fcl
tool_type: FclFloatArray
Label: "AreaGain_ib3_b900_apau_fixP0_fixN1_fit-0p5-7p5"
Unit: "ke/(ADC-count)/tick"
Values: [
  0.2458702, 0.1949961, 0.1949987, 0.1949987, 0.1949987, ...
```

- `calib/` (310) and `pedestal/` (6) are pure float arrays, 50–264 lines each.
- `rundata/` (143) are 8-line run-condition scalars (`gain`, `shaping`,
  `hvfrac`) that differ between files by a couple of numbers.

Every file in a group embeds to nearly the same vector. They add 456 chunks
that can outrank real answers on any query mentioning calibration or gain,
while carrying no information a reader could use. Worst possible
value-to-noise ratio in the corpus.

### `prod*.fcl` — 752 files

Mass-production job configs, overwhelmingly under `dunesw/fcl/*/gen/`.
Combinatorial permutations of the same job:

```
prodaddgenie_bdm_b1p5_m05_01b_dune10kt_1x2x6_refactored.fcl
prodaddgenie_bdm_b1p5_m05_02a_dune10kt.fcl
prodaddgenie_bdm_b1p5_m05_02a_dune10kt_1x2x6.fcl
```

Each is 7–12 lines: an `#include` of its parent, a services table swap, an
output filename. The distinct content is one `fileList` line. 752 files
expressing maybe a dozen real ideas.

**Scoped by filename prefix, not directory**, deliberately: the 89 non-`prod*`
files under `gen/` are genuine generator configs (`MUSUN_dune10kt_1x2x6.fcl`,
`workflow_radiological_decay0_dune10kt.fcl`) and are kept.

### `test/` — 395 files

Test directories across all repos. Fixtures and assertions describe what the
code must not do, not what it does or how to call it. Reconsider if the
benchmark ever needs to reason about test coverage.

### `larreco/larreco/Genfit/` — 60 files

Vendored GenFit track-fitting library. Third-party code, not DUNE or LArSoft
authored. Out of scope for a DUNE codebase assistant, and its presence would
let generic Kalman-filter questions pull answers that aren't DUNE's
implementation.

## 3. Dropped repo: `larrecoalg`

`LArSoft/larrecoalg` on GitHub is an empty skeleton — 8 files, zero source,
just `CMakeLists.txt`, `LICENSE`, and `ups/product_deps`. Both `develop` and
`main` sit on the same commit. Its README even reads `# lardataalg`, a
copy-paste never fixed. Nothing else in Tier 1 declares a dependency on it.

The reconstruction algorithms it was supposed to hold are in
**`larreco/larreco/RecoAlg/` (293 files)**, already covered by the larreco
clone. Tier 1 is therefore **24 repos**, not 25, and nothing is lost.

## 4. Not an exclusion: the `.h` bug this surfaced

Before this pass, `DISPATCH` contained `.H` but not `.h`. Across Tier 1 there
are 1,636 `.h` files and **zero** `.H` files, so every lowercase header was
skipped at the `DISPATCH.get(ext) is None` check — silently, because the
whole-file fallback only fires for files that reach a parser.

Verified against the live index: `chunks.jsonl` held 576 unique files and no
headers at all.

For a C++ codebase this removed the class declarations, public method
signatures, and interface doc comments — the exact content most queries want.
`parse_cpp` handles headers correctly; `lardataobj/RecoBase/Hit.h` yields 22
symbol-level chunks (`recob::Hit::PeakTimeMinusRMS`, `recob::Hit::View`, …).

Fixed by adding `.h`, `.tcc`, and `.c++` to the allowlist.

## Reproducing these counts

```bash
python extract.py <repo_root>
```

The extraction summary reports per-language file and chunk counts plus
whole-file fallbacks. Excluded paths are counted and printed separately, so a
pattern that silently stops matching after an upstream reorganisation shows up
as a zero rather than passing unnoticed.
