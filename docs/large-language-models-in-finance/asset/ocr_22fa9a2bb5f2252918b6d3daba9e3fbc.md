```
# Stronger preference if numerical accuracy differs
if self.has_calculations(preferred) or self.has_calculations(rejected):
    acc_diff = abs(
        self.get_numerical_accuracy(preferred) -
        self.get_numerical_accuracy(rejected))
    strength *= (1 + acc_diff)

# Stronger preference if compliance differs
if self.has_compliance_content(preferred) or self.has_compliance_content
    (rejected):
    comp_diff = abs(
        self.get_compliance_score(preferred) -
        self.get_compliance_score(rejected))
    strength *= (1 + 0.5 * comp_diff)

preference_strengths.append(strength)

augmented_batch['preference_strength'] = torch.tensor(
    preference_strengths)

return augmented_batch
```

![](bbox=[0.14, 0.47, 0.86, 0.68])

Figure 9.1: Three routes from preferences to an aligned policy.

They differ in what machinery they need — PPO wants a reward model and a value network, DPO discards both, GRPO keeps the RL loop but replaces the value network with a group-relative baseline — and they are usually compared on that axis alone. In finance the axis that matters is the one on the right: none of the three has any notion of a risk limit or a suitability rule, so the constraint set is not a property of the algorithm and must be imposed outside it.