```
It is a heuristic, and it is off by default.
"""
# ---- PHASE 1: freeze the behaviour policy for the WHOLE batch ----
# A snapshot taken inside the prompt loop is a snapshot per GROUP, not
# per batch: once the optimizer steps for prompt 1, prompt 2's samples
# come from a policy that no longer equals the one whose log-probs sit
# in the denominator of the ratio. The importance weight is then against
# a policy that never generated anything, and the clip range no longer
# means what it says. Snapshot once, generate everything, THEN update.
pi_old = copy.deepcopy(self.model).eval()
for p in pi_old.parameters():
    p.requires_grad_(False)

batch = []
with torch.no_grad():
    for prompt in training_prompts:
        responses = self.generate_group_samples(
            prompt, self.config.group_size, policy=pi_old)
        advantages, _ = self.compute_group_advantages(prompt, responses)
        batch.append({
            "prompt": prompt,
            "responses": responses,
            "advantages": advantages,
            "old_lp": [pi_old.get_log_probs(prompt, r) for r in responses],
            "ref_lp": [self.ref_model.get_log_probs(prompt, r)
                       for r in responses],
        })

# ---- PHASE 2: optimize the current policy on that frozen batch ----
total_loss, num_updates = 0.0, 0
for _ in range(self.config.ppo_epochs):
    self.optimizer.zero_grad()
    batch_loss = 0.0
    n_terms = 0

    for g in batch:
        idxs = range(len(g["responses"]))
        if self.config.use_advantage_filtering:
            # Off by default: discarding negative advantages throws away
            # exactly the half of the group carrying the down-gradient.
            idxs = [i for i in idxs if g["advantages"][i] > 0]

        for i in idxs:
            lp = self.model.get_log_probs(g["prompt"], g["responses"][i])
            adv = g["advantages"][i]

            # Ratio against the FROZEN behaviour policy.
            ratio = torch.exp(lp - g["old_lp"][i])
            surr1 = ratio * adv
```