| Violation | Clean IC | Violation IC | ΔIC | tΔ |
|---|---|---|---|---|
| A1: full-sample supervised oracle | 0.0665 | 0.1721 | +0.1056 | 15.78 |
| A2: matched-size future oracle | 0.0665 | 0.1379 | +0.0714 | 8.11 |
| B: labels not yet realized | 0.0665 | 0.0676 | +0.0011 | 0.93 |
| C: full-sample normalization | 0.0665 | 0.0665 | 0.0000 | 0.00 |
| D: timestamp fiction | 0.0665 | 0.0672 | +0.0007 | 0.36 |

Table 13.5: [Evidence level D] Author-generated synthetic illustration — and now reproducible.

These numbers measure a leakage channel under controlled conditions. They are not a measurement of any market, and by the ladder of Section 12.6.1 they remain Status D. What has changed is that they can now be checked.

Table 13.5 shows the fixed-seed anatomy, and three of the five rows say something the design did not anticipate.

The full-sample supervised oracle (A1) raises mean rank IC from 0.0665 to 0.1721: it more than doubles the signal, with a Newey–West _t_ of 15.78 and a block-bootstrap interval of [0.0930, 0.1186] nowhere near zero. That is the easy result. The instructive one is A2.

**A2 does not collapse.** Matching the training count to the clean protocol — so that the only remaining difference is _when_ the labels came from — still leaves a gap of +0.0714 with _t_ = 8.11. The advantage therefore cannot be narrated as “the oracle simply saw more data.” It saw future data, and future data is worth more than an equal quantity of past data. That is the whole argument of this section reduced to one row of a table, and it is the row to ask about first when someone reports that their model improved after “retraining on the full history.”

**Full-sample normalization (C) has exactly zero effect — and that is not good news.** The ΔIC is 0.0000 to four places, with _t_ = 0.00. The reason is arithmetic: a global affine rescaling of a score cannot change the _cross-sectional ranks_ of that score on any given day, and rank IC depends only on ranks. The leakage channel is entirely real — the normalizer has seen the whole sample — but this particular metric is structurally blind to it. Do not read the zero as an all-clear. Read it as a warning that a metric can be incapable of detecting a violation that is unambiguously present, which is exactly why the audit in this section is stated over _artifacts_ rather than over test statistics. A clean number is not a clean pipeline.

Label admission before realization (B, +0.0011) and timestamp fiction (D, +0.0007) are small in this calibration. Both are genuine channels; both are weak here because the simulated post-cutoff articles carry the same distribution as the pre-cutoff ones, so the extra coverage that D manufactures is mostly noise. In a real corpus, where news arriving after the close is systematically different from news arriving before it, D is a far larger channel. That sensitivity is a property of the calibration, not of the method, and it is why this table measures _mechanism_ rather than magnitude.