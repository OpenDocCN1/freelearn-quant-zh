and queries that are genuinely ambiguous or that draw on contradictory information from different sections of the same document:

+   - Very long documents
+   - Multiple entities with similar names
+   - Ambiguous or underspecified queries
+   - Contradictory information from multiple sources

Failures under stress testing do not necessarily mean a model should not be deployed; they define the boundary conditions under which it operates reliably and those under which it requires human oversight. Documenting these boundaries explicitly is itself a governance deliverable.

The final step is to observe how the model behaves not in controlled tests, but in real usage.

# 3.8.4 A/B testing in production

Offline evaluation on a held-out test set measures potential; A/B testing in production measures actual impact. The two are often surprisingly different. A model that scores several percentage points higher on a benchmark may produce outputs that real users find less useful, trust less, or override more frequently—signals that only become visible when the model is interacting with genuine tasks and genuine users.

The standard approach routes a small fraction of live traffic—typically 5–10%—to the new model while continuing to serve the baseline model to the remainder. Crucially, every request and response is logged alongside any available signal of user satisfaction (explicit feedback, override rates, follow-up queries). This creates the evidence base needed to determine whether the benchmark improvement translates into a real-world improvement before a full rollout is committed:

```
```
import hashlib

def stable_bucket(session_id):
    digest = hashlib.sha256(str(session_id).encode()).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF

# Route 10% of sessions, not individual queries, to the new model.
if stable_bucket(user_session_id) < 0.10:
    model_version = "new"
    response = new_fine_tuned_model(query)
else:
    model_version = "baseline"
    response = baseline_model(query)

# Log metrics with the sticky session assignment.
log_metrics(model_version, user_session_id, query, response, user_feedback)
```
```