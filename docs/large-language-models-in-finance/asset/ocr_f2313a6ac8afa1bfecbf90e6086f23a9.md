engine must determine which products, disclosures, conflicts, fees, assumptions, and warnings apply. The recommendation, evidence, optimizer inputs, and generated explanation should then be reviewed and approved by an authorized human advisor whenever the activity constitutes regulated advice. The LLM may support information gathering and explanation, but it should not make the final suitability determination, suppress required disclosures, or submit a transaction on its own [U.S. Securities and Exchange Commission, 2019].

A clean division of labor is shown below:

+   - validated data services maintain the client profile and entitlements;
+   - deterministic rules enforce jurisdictional and product constraints;
+   - tested quantitative models produce portfolio analytics;
+   - the LLM drafts a plain-language explanation grounded in those inputs;
+   - an authorized professional approves, modifies, or rejects the draft; and
+   - a separate transaction system executes only an approved instruction.

The workflow in Listing 8.4 makes the approval boundary explicit.

# Listing 8.4: Bounded advisory workflow with human approval

```
def prepare_advisory_draft(client_id, request, services):
    profile = services.client_profile.get_validated(client_id)
    policy = services.policy_engine.resolve(
        jurisdiction=profile.jurisdiction,
        activity="investment_advice",
    )

    suitability = services.suitability.evaluate(profile, request, policy)
    if not suitability.eligible:
        return route_to_advisor(suitability.reasons)

    analytics = services.portfolio_model.evaluate(
        holdings=profile.holdings,
        objectives=profile.objectives,
        constraints=suitability.constraints,
    )

    explanation = services.llm.draft_explanation(
        approved_facts=analytics,
        required_disclosures=policy.disclosures,
        prohibited_claims=policy.prohibited_claims,
    )

    return {
        "status": "pending_human_approval",
        "analytics": analytics,
        "explanation": explanation,
        "disclosures": policy.disclosures,
```