Calibration is equally important. For a classifier that outputs probabilities \(p_i\), expected calibration error can be estimated by binning predictions into confidence bins \(B_m\) and computing

```
\(\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{n} |\text{acc}(B_m) - \text{conf}(B_m)|.\) (3.2)
```

In finance, calibration is often more important than raw accuracy. A sentiment model that is correct 80% of the time but cannot distinguish 55% confidence from 95% confidence is hard to use in position sizing, escalation, or client-facing disclosure. Calibration should therefore be reported alongside accuracy, F1, drawdown-sensitive downstream metrics, and human-review outcomes [Guo et al., 2017].

The implementation pattern below is intentionally small, but it encodes the production rule that matters most: split by availability time, validate required fields, and fail when the split is empty or temporally inconsistent.

# Listing 3.1: Auditable temporal split for financial fine-tuning

```
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Sequence

@dataclass(frozen=True)
class LabeledExample:
    example_id: str
    text: str
    label: str
    available_at: datetime
    group: str | None = None

@dataclass(frozen=True)
class TemporalSplit:
    train: tuple[LabeledExample, ...]
    valid: tuple[LabeledExample, ...]
    test: tuple[LabeledExample, ...]

def make_temporal_split(
    examples: Iterable[LabeledExample],
    train_end: datetime,
    valid_end: datetime,
) -> TemporalSplit:
    if train_end >= valid_end:
        raise ValueError("train_end must be strictly before valid_end")
    ordered = sorted(examples, key=lambda z: (z.available_at, z.example_id))
    if any(not z.text.strip() for z in ordered):
        raise ValueError("empty text example found")
    train = tuple(z for z in ordered if z.available_at <= train_end)
```