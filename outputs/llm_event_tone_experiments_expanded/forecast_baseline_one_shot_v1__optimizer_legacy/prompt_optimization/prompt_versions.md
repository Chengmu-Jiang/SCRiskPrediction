# Prompt Versions

## Version 0

- Guidance hash: `5677be43abd1`
- Forecast prompt path: `configs\llm_tone\prompts\forecast\baseline_one_shot_v1.yaml`
- Optimizer prompt path: `configs\llm_tone\prompts\optimizer\legacy.yaml`
- Val RMSE: `3.2507650172726463`
- Val MAE: `2.5798465699787774`
- Test RMSE: `2.7722865144994913`
- Test MAE: `2.3777182727439943`

### Guidance

(original prompt; no extra guidance)

### Prompt Template

```text
You are forecasting the next effective news event for a product/topic.

Format-only example:
The following anonymized example only illustrates the expected input-output style, level of detail, and length.
Do not copy its topic, actors, countries, sentiment direction, risk severity, wording, or event type.

Example Topic/Product: Industrial semiconductor components
Example HS4 code: 0000
Example HS4 label: Example input category
Example Last observed effective news date: 2024-04-05
Example Prediction target: the next effective news date after 2024-04-05, not the next calendar day.

Example historical compressed daily news sequence:
[1] Date: 2024-04-01
Summary: A government intervention in a semiconductor company increased uncertainty over ownership, exports, and supply continuity. Downstream manufacturers began monitoring possible component shortages.

[2] Date: 2024-04-02
Summary: Officials defended the intervention as necessary to prevent technology transfer, while the company's overseas parent responded with export restrictions. Packaging operations and wafer shipments became points of dispute.

[3] Date: 2024-04-03
Summary: Industry groups urged policymakers to resolve the dispute through dialogue. Automakers and suppliers warned that prolonged restrictions could create renewed shortages and force longer-term sourcing changes.

[4] Date: 2024-04-04
Summary: Commentators framed the company as exposed to broader trade tensions. The report emphasized geopolitical pressure, alliance politics, and the difficulty of restoring trust among regulators and corporate owners.

[5] Date: 2024-04-05
Summary: The parent company invited court-appointed representatives to discuss control and supply restoration. The local business said it saw limited evidence that production flows would resume quickly.

Example generated future-event description:
A commerce ministry urged the foreign government to support direct talks between the semiconductor company's local unit and its overseas parent. Officials said diplomatic and economic agencies should help implement a consultation agreement and encourage company representatives to travel for negotiations. The parent company described a recent invitation to independent directors and trustees as a constructive step toward restoring global supply-chain stability. The report suggested a cautious thaw after weeks of accusations, but the underlying control dispute and wafer-flow disruptions remained unresolved, leaving downstream manufacturers exposed to renewed shortages if negotiations stalled.

End of format-only example.

Now forecast the actual case.

Topic/Product: {topic}
HS4 code: {hs_code_4d}
HS4 label: {hs_short_label}
Last observed effective news date: {last_history_date}
Prediction target: the next effective news date after the last observed effective news date, not the next calendar day.

Historical compressed daily news sequence:
{history_text}

Task:
Generate a plausible future event description for the next effective news date.

Constraints:
- Use only the actual historical sequence as evidence.
- Use the example only for output format, information density, and length.
- Do not copy the example's topic, actors, countries, sentiment direction, risk severity, wording, or event type.
- Focus on the likely event narrative and tone drivers.
- Do not output ArticleScore, AvgTone, labels, bullet points, or explanations.
- Do not include hidden reasoning, chain-of-thought, analysis notes, or <think> tags.
- Use 80 to 180 words.
- Return only the generated future-event description.
```

## Version 1

- Guidance hash: `78ec90f792c1`
- Forecast prompt path: `configs\llm_tone\prompts\forecast\baseline_one_shot_v1.yaml`
- Optimizer prompt path: `configs\llm_tone\prompts\optimizer\legacy.yaml`
- Val RMSE: `3.2130419751757993`
- Val MAE: `2.6024874534624014`
- Test RMSE: `2.555725501442126`
- Test MAE: `2.123217440056671`

### Guidance

- Avoid exaggerating conflict escalation or negative outcomes; weigh evidence of de-escalation, constructive talks, or operational workarounds that appear in the historical sequence.
- Do not generate speculative future events not supported by the historical trend; prefer ambiguous or moderate language when the trajectory is uncertain and the true news often conveys tempered or mixed signals.

### Prompt Template

```text
You are forecasting the next effective news event for a product/topic.

Format-only example:
The following anonymized example only illustrates the expected input-output style, level of detail, and length.
Do not copy its topic, actors, countries, sentiment direction, risk severity, wording, or event type.

Example Topic/Product: Industrial semiconductor components
Example HS4 code: 0000
Example HS4 label: Example input category
Example Last observed effective news date: 2024-04-05
Example Prediction target: the next effective news date after 2024-04-05, not the next calendar day.

Example historical compressed daily news sequence:
[1] Date: 2024-04-01
Summary: A government intervention in a semiconductor company increased uncertainty over ownership, exports, and supply continuity. Downstream manufacturers began monitoring possible component shortages.

[2] Date: 2024-04-02
Summary: Officials defended the intervention as necessary to prevent technology transfer, while the company's overseas parent responded with export restrictions. Packaging operations and wafer shipments became points of dispute.

[3] Date: 2024-04-03
Summary: Industry groups urged policymakers to resolve the dispute through dialogue. Automakers and suppliers warned that prolonged restrictions could create renewed shortages and force longer-term sourcing changes.

[4] Date: 2024-04-04
Summary: Commentators framed the company as exposed to broader trade tensions. The report emphasized geopolitical pressure, alliance politics, and the difficulty of restoring trust among regulators and corporate owners.

[5] Date: 2024-04-05
Summary: The parent company invited court-appointed representatives to discuss control and supply restoration. The local business said it saw limited evidence that production flows would resume quickly.

Example generated future-event description:
A commerce ministry urged the foreign government to support direct talks between the semiconductor company's local unit and its overseas parent. Officials said diplomatic and economic agencies should help implement a consultation agreement and encourage company representatives to travel for negotiations. The parent company described a recent invitation to independent directors and trustees as a constructive step toward restoring global supply-chain stability. The report suggested a cautious thaw after weeks of accusations, but the underlying control dispute and wafer-flow disruptions remained unresolved, leaving downstream manufacturers exposed to renewed shortages if negotiations stalled.

End of format-only example.

Now forecast the actual case.

Topic/Product: {topic}
HS4 code: {hs_code_4d}
HS4 label: {hs_short_label}
Last observed effective news date: {last_history_date}
Prediction target: the next effective news date after the last observed effective news date, not the next calendar day.

Historical compressed daily news sequence:
{history_text}

Task:
Generate a plausible future event description for the next effective news date.

Constraints:
- Use only the actual historical sequence as evidence.
- Use the example only for output format, information density, and length.
- Do not copy the example's topic, actors, countries, sentiment direction, risk severity, wording, or event type.
- Focus on the likely event narrative and tone drivers.
- Do not output ArticleScore, AvgTone, labels, bullet points, or explanations.
- Do not include hidden reasoning, chain-of-thought, analysis notes, or <think> tags.
- Use 80 to 180 words.
- Return only the generated future-event description.

Additional tone-calibration guidance learned from training feedback:
- Avoid exaggerating conflict escalation or negative outcomes; weigh evidence of de-escalation, constructive talks, or operational workarounds that appear in the historical sequence.
- Do not generate speculative future events not supported by the historical trend; prefer ambiguous or moderate language when the trajectory is uncertain and the true news often conveys tempered or mixed signals.
```

## Version 2

- Guidance hash: `65868731f27a`
- Forecast prompt path: `configs\llm_tone\prompts\forecast\baseline_one_shot_v1.yaml`
- Optimizer prompt path: `configs\llm_tone\prompts\optimizer\legacy.yaml`
- Val RMSE: `3.185242701055776`
- Val MAE: `2.53604772376867`
- Test RMSE: `2.5602666176731494`
- Test MAE: `2.0847394164087585`

### Guidance

- Avoid exaggerating conflict escalation or negative outcomes; weigh evidence of de-escalation, constructive talks, or operational workarounds that appear in the historical sequence.
- Do not generate speculative future events not supported by the historical trend; prefer ambiguous or moderate language when the trajectory is uncertain and the true news often conveys tempered or mixed signals.
- Do not infer de-escalation or constructive resolution unless the historical sequence contains explicit, verifiable steps toward a settlement; prefer ambiguous or moderate language when the trajectory remains uncertain or mixed.
- Avoid generating specific future events (e.g., court rulings, official exemptions, formal agreements) that are not directly supported by the most recent historical entries; instead, describe ongoing dynamics in more general and cautious terms.
- When true news frequently shifts to unrelated topics or negative developments, the generated description should not assume continuity of the previous narrative; consider that the story may diverge or stall, and keep forecasts grounded in the latest updates.

### Prompt Template

```text
You are forecasting the next effective news event for a product/topic.

Format-only example:
The following anonymized example only illustrates the expected input-output style, level of detail, and length.
Do not copy its topic, actors, countries, sentiment direction, risk severity, wording, or event type.

Example Topic/Product: Industrial semiconductor components
Example HS4 code: 0000
Example HS4 label: Example input category
Example Last observed effective news date: 2024-04-05
Example Prediction target: the next effective news date after 2024-04-05, not the next calendar day.

Example historical compressed daily news sequence:
[1] Date: 2024-04-01
Summary: A government intervention in a semiconductor company increased uncertainty over ownership, exports, and supply continuity. Downstream manufacturers began monitoring possible component shortages.

[2] Date: 2024-04-02
Summary: Officials defended the intervention as necessary to prevent technology transfer, while the company's overseas parent responded with export restrictions. Packaging operations and wafer shipments became points of dispute.

[3] Date: 2024-04-03
Summary: Industry groups urged policymakers to resolve the dispute through dialogue. Automakers and suppliers warned that prolonged restrictions could create renewed shortages and force longer-term sourcing changes.

[4] Date: 2024-04-04
Summary: Commentators framed the company as exposed to broader trade tensions. The report emphasized geopolitical pressure, alliance politics, and the difficulty of restoring trust among regulators and corporate owners.

[5] Date: 2024-04-05
Summary: The parent company invited court-appointed representatives to discuss control and supply restoration. The local business said it saw limited evidence that production flows would resume quickly.

Example generated future-event description:
A commerce ministry urged the foreign government to support direct talks between the semiconductor company's local unit and its overseas parent. Officials said diplomatic and economic agencies should help implement a consultation agreement and encourage company representatives to travel for negotiations. The parent company described a recent invitation to independent directors and trustees as a constructive step toward restoring global supply-chain stability. The report suggested a cautious thaw after weeks of accusations, but the underlying control dispute and wafer-flow disruptions remained unresolved, leaving downstream manufacturers exposed to renewed shortages if negotiations stalled.

End of format-only example.

Now forecast the actual case.

Topic/Product: {topic}
HS4 code: {hs_code_4d}
HS4 label: {hs_short_label}
Last observed effective news date: {last_history_date}
Prediction target: the next effective news date after the last observed effective news date, not the next calendar day.

Historical compressed daily news sequence:
{history_text}

Task:
Generate a plausible future event description for the next effective news date.

Constraints:
- Use only the actual historical sequence as evidence.
- Use the example only for output format, information density, and length.
- Do not copy the example's topic, actors, countries, sentiment direction, risk severity, wording, or event type.
- Focus on the likely event narrative and tone drivers.
- Do not output ArticleScore, AvgTone, labels, bullet points, or explanations.
- Do not include hidden reasoning, chain-of-thought, analysis notes, or <think> tags.
- Use 80 to 180 words.
- Return only the generated future-event description.

Additional tone-calibration guidance learned from training feedback:
- Avoid exaggerating conflict escalation or negative outcomes; weigh evidence of de-escalation, constructive talks, or operational workarounds that appear in the historical sequence.
- Do not generate speculative future events not supported by the historical trend; prefer ambiguous or moderate language when the trajectory is uncertain and the true news often conveys tempered or mixed signals.
- Do not infer de-escalation or constructive resolution unless the historical sequence contains explicit, verifiable steps toward a settlement; prefer ambiguous or moderate language when the trajectory remains uncertain or mixed.
- Avoid generating specific future events (e.g., court rulings, official exemptions, formal agreements) that are not directly supported by the most recent historical entries; instead, describe ongoing dynamics in more general and cautious terms.
- When true news frequently shifts to unrelated topics or negative developments, the generated description should not assume continuity of the previous narrative; consider that the story may diverge or stall, and keep forecasts grounded in the latest updates.
```

## Version 3

- Guidance hash: `4a7a2b1ad9f2`
- Forecast prompt path: `configs\llm_tone\prompts\forecast\baseline_one_shot_v1.yaml`
- Optimizer prompt path: `configs\llm_tone\prompts\optimizer\legacy.yaml`
- Val RMSE: `3.163493132445179`
- Val MAE: `2.621728796185652`
- Test RMSE: `2.6043077177664555`
- Test MAE: `2.094020665363234`

### Guidance

- Avoid exaggerating conflict escalation or negative outcomes; weigh evidence of de-escalation, constructive talks, or operational workarounds that appear in the historical sequence.
- Do not generate speculative future events not supported by the historical trend; prefer ambiguous or moderate language when the trajectory is uncertain and the true news often conveys tempered or mixed signals.
- Do not infer de-escalation or constructive resolution unless the historical sequence contains explicit, verifiable steps toward a settlement; prefer ambiguous or moderate language when the trajectory remains uncertain or mixed.
- Avoid generating specific future events (e.g., court rulings, official exemptions, formal agreements) that are not directly supported by the most recent historical entries; instead, describe ongoing dynamics in more general and cautious terms.
- When true news frequently shifts to unrelated topics or negative developments, the generated description should not assume continuity of the previous narrative; consider that the story may diverge or stall, and keep forecasts grounded in the latest updates.
- Do not project a single linear storyline (e.g., legal battles, diplomatic negotiations) across multiple future updates; instead, treat each prediction as an independent point in time and base it strictly on the most recent historical entry, acknowledging that the news may pivot to an entirely different subtopic or reveal a contradictory development.
- Avoid inferring specific future events—such as scheduled hearings, ministerial statements, joint industry actions, or licensing expansions—unless the latest historical entry explicitly states such an event is imminent; prefer describing the current unresolved state with ambiguous language that leaves open both continuation and sudden divergence.

### Prompt Template

```text
You are forecasting the next effective news event for a product/topic.

Format-only example:
The following anonymized example only illustrates the expected input-output style, level of detail, and length.
Do not copy its topic, actors, countries, sentiment direction, risk severity, wording, or event type.

Example Topic/Product: Industrial semiconductor components
Example HS4 code: 0000
Example HS4 label: Example input category
Example Last observed effective news date: 2024-04-05
Example Prediction target: the next effective news date after 2024-04-05, not the next calendar day.

Example historical compressed daily news sequence:
[1] Date: 2024-04-01
Summary: A government intervention in a semiconductor company increased uncertainty over ownership, exports, and supply continuity. Downstream manufacturers began monitoring possible component shortages.

[2] Date: 2024-04-02
Summary: Officials defended the intervention as necessary to prevent technology transfer, while the company's overseas parent responded with export restrictions. Packaging operations and wafer shipments became points of dispute.

[3] Date: 2024-04-03
Summary: Industry groups urged policymakers to resolve the dispute through dialogue. Automakers and suppliers warned that prolonged restrictions could create renewed shortages and force longer-term sourcing changes.

[4] Date: 2024-04-04
Summary: Commentators framed the company as exposed to broader trade tensions. The report emphasized geopolitical pressure, alliance politics, and the difficulty of restoring trust among regulators and corporate owners.

[5] Date: 2024-04-05
Summary: The parent company invited court-appointed representatives to discuss control and supply restoration. The local business said it saw limited evidence that production flows would resume quickly.

Example generated future-event description:
A commerce ministry urged the foreign government to support direct talks between the semiconductor company's local unit and its overseas parent. Officials said diplomatic and economic agencies should help implement a consultation agreement and encourage company representatives to travel for negotiations. The parent company described a recent invitation to independent directors and trustees as a constructive step toward restoring global supply-chain stability. The report suggested a cautious thaw after weeks of accusations, but the underlying control dispute and wafer-flow disruptions remained unresolved, leaving downstream manufacturers exposed to renewed shortages if negotiations stalled.

End of format-only example.

Now forecast the actual case.

Topic/Product: {topic}
HS4 code: {hs_code_4d}
HS4 label: {hs_short_label}
Last observed effective news date: {last_history_date}
Prediction target: the next effective news date after the last observed effective news date, not the next calendar day.

Historical compressed daily news sequence:
{history_text}

Task:
Generate a plausible future event description for the next effective news date.

Constraints:
- Use only the actual historical sequence as evidence.
- Use the example only for output format, information density, and length.
- Do not copy the example's topic, actors, countries, sentiment direction, risk severity, wording, or event type.
- Focus on the likely event narrative and tone drivers.
- Do not output ArticleScore, AvgTone, labels, bullet points, or explanations.
- Do not include hidden reasoning, chain-of-thought, analysis notes, or <think> tags.
- Use 80 to 180 words.
- Return only the generated future-event description.

Additional tone-calibration guidance learned from training feedback:
- Avoid exaggerating conflict escalation or negative outcomes; weigh evidence of de-escalation, constructive talks, or operational workarounds that appear in the historical sequence.
- Do not generate speculative future events not supported by the historical trend; prefer ambiguous or moderate language when the trajectory is uncertain and the true news often conveys tempered or mixed signals.
- Do not infer de-escalation or constructive resolution unless the historical sequence contains explicit, verifiable steps toward a settlement; prefer ambiguous or moderate language when the trajectory remains uncertain or mixed.
- Avoid generating specific future events (e.g., court rulings, official exemptions, formal agreements) that are not directly supported by the most recent historical entries; instead, describe ongoing dynamics in more general and cautious terms.
- When true news frequently shifts to unrelated topics or negative developments, the generated description should not assume continuity of the previous narrative; consider that the story may diverge or stall, and keep forecasts grounded in the latest updates.
- Do not project a single linear storyline (e.g., legal battles, diplomatic negotiations) across multiple future updates; instead, treat each prediction as an independent point in time and base it strictly on the most recent historical entry, acknowledging that the news may pivot to an entirely different subtopic or reveal a contradictory development.
- Avoid inferring specific future events—such as scheduled hearings, ministerial statements, joint industry actions, or licensing expansions—unless the latest historical entry explicitly states such an event is imminent; prefer describing the current unresolved state with ambiguous language that leaves open both continuation and sudden divergence.
```
