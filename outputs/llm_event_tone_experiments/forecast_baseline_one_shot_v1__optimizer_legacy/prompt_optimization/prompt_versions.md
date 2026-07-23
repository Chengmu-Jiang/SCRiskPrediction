# Prompt Versions

## Version 0

- Guidance hash: `5677be43abd1`
- Forecast prompt path: `configs\llm_tone\prompts\forecast\baseline_one_shot_v1.yaml`
- Optimizer prompt path: `configs\llm_tone\prompts\optimizer\legacy.yaml`
- Val RMSE: `3.322848480180471`
- Val MAE: `2.7292258730764236`
- Test RMSE: `2.797159099837498`
- Test MAE: `2.3675723942117757`

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

- Guidance hash: `74758334a78c`
- Forecast prompt path: `configs\llm_tone\prompts\forecast\baseline_one_shot_v1.yaml`
- Optimizer prompt path: `configs\llm_tone\prompts\optimizer\legacy.yaml`
- Val RMSE: `3.569540597975105`
- Val MAE: `2.817849453175053`
- Test RMSE: `3.577484625279321`
- Test MAE: `3.1253279982670303`

### Guidance

- Avoid assuming direct continuation of the most recent event arc; consider that the next article may shift to a different stakeholder perspective, broader industry analysis, or a completely unrelated topic.
- Ground tone in the full range of possibilities: both escalation and de-escalation, and do not default to high risk or conflict severity unless strongly supported by historical patterns; account for potential positive developments such as supply restoration or strategic adaptations.

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
- Avoid assuming direct continuation of the most recent event arc; consider that the next article may shift to a different stakeholder perspective, broader industry analysis, or a completely unrelated topic.
- Ground tone in the full range of possibilities: both escalation and de-escalation, and do not default to high risk or conflict severity unless strongly supported by historical patterns; account for potential positive developments such as supply restoration or strategic adaptations.
```

## Version 2

- Guidance hash: `20e72efbd7b5`
- Forecast prompt path: `configs\llm_tone\prompts\forecast\baseline_one_shot_v1.yaml`
- Optimizer prompt path: `configs\llm_tone\prompts\optimizer\legacy.yaml`
- Val RMSE: `3.5856000288376366`
- Val MAE: `2.8650879250479333`
- Test RMSE: `3.278327886245404`
- Test MAE: `2.811294041251474`

### Guidance

- Avoid assuming direct continuation of the most recent event arc; consider that the next article may shift to a different stakeholder perspective, broader industry analysis, or a completely unrelated topic.
- Ground tone in the full range of possibilities: both escalation and de-escalation, and do not default to high risk or conflict severity unless strongly supported by historical patterns; account for potential positive developments such as supply restoration or strategic adaptations.
- Explicitly account for the possibility that the next article may be entirely unrelated to the historical topic thread (e.g., local politics, company-specific updates, or other industries) and avoid defaulting to the most salient crisis narrative.
- Do not generate specific future events (e.g., rulings, investments, production changes) unless historical news provides strong, multi-source evidence of a concrete timeline; prefer generic updates, statements, or diplomatic reactions when evidence is thin.
- Balance the tone between escalation and de-escalation: when historical news shows ongoing tensions, the true next article may be an escalation (e.g., official rebuke) rather than a compromise or solution, and vice versa for resolutions.

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
- Avoid assuming direct continuation of the most recent event arc; consider that the next article may shift to a different stakeholder perspective, broader industry analysis, or a completely unrelated topic.
- Ground tone in the full range of possibilities: both escalation and de-escalation, and do not default to high risk or conflict severity unless strongly supported by historical patterns; account for potential positive developments such as supply restoration or strategic adaptations.
- Explicitly account for the possibility that the next article may be entirely unrelated to the historical topic thread (e.g., local politics, company-specific updates, or other industries) and avoid defaulting to the most salient crisis narrative.
- Do not generate specific future events (e.g., rulings, investments, production changes) unless historical news provides strong, multi-source evidence of a concrete timeline; prefer generic updates, statements, or diplomatic reactions when evidence is thin.
- Balance the tone between escalation and de-escalation: when historical news shows ongoing tensions, the true next article may be an escalation (e.g., official rebuke) rather than a compromise or solution, and vice versa for resolutions.
```

## Version 3

- Guidance hash: `cb52274b04ea`
- Forecast prompt path: `configs\llm_tone\prompts\forecast\baseline_one_shot_v1.yaml`
- Optimizer prompt path: `configs\llm_tone\prompts\optimizer\legacy.yaml`
- Val RMSE: `3.588737640736709`
- Val MAE: `2.9058343403208893`
- Test RMSE: `3.58092125446698`
- Test MAE: `3.0146091067312044`

### Guidance

- Avoid assuming direct continuation of the most recent event arc; consider that the next article may shift to a different stakeholder perspective, broader industry analysis, or a completely unrelated topic.
- Ground tone in the full range of possibilities: both escalation and de-escalation, and do not default to high risk or conflict severity unless strongly supported by historical patterns; account for potential positive developments such as supply restoration or strategic adaptations.
- Explicitly account for the possibility that the next article may be entirely unrelated to the historical topic thread (e.g., local politics, company-specific updates, or other industries) and avoid defaulting to the most salient crisis narrative.
- Do not generate specific future events (e.g., rulings, investments, production changes) unless historical news provides strong, multi-source evidence of a concrete timeline; prefer generic updates, statements, or diplomatic reactions when evidence is thin.
- Balance the tone between escalation and de-escalation: when historical news shows ongoing tensions, the true next article may be an escalation (e.g., official rebuke) rather than a compromise or solution, and vice versa for resolutions.
- When recent historical news is dominated by a single crisis narrative, explicitly consider that the next article may be completely unrelated (e.g., local politics, company-specific earnings, crime) and assign a neutral tone with no escalation.
- Avoid generating any specific future events (e.g., consortium investments, court rulings, diplomatic missions, production milestones) unless multiple historical sources provide concrete, multi-day evidence of a scheduled timeline; prefer generic statements like 'industry sources commented' or 'no new developments'.
- If the historical news shows a mature dispute with repeated stakeholder statements but no resolution, do not manufacture new diplomatic moves or operational shifts: the next article may simply be a minor update from a single company or an entirely different topic.
- When the crisis involves multiple governments and industry bodies, the true next article often narrows to a single company's isolated action (e.g., a local chip development plan, a CEO interview) rather than broad coordination or further escalation.

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
- Avoid assuming direct continuation of the most recent event arc; consider that the next article may shift to a different stakeholder perspective, broader industry analysis, or a completely unrelated topic.
- Ground tone in the full range of possibilities: both escalation and de-escalation, and do not default to high risk or conflict severity unless strongly supported by historical patterns; account for potential positive developments such as supply restoration or strategic adaptations.
- Explicitly account for the possibility that the next article may be entirely unrelated to the historical topic thread (e.g., local politics, company-specific updates, or other industries) and avoid defaulting to the most salient crisis narrative.
- Do not generate specific future events (e.g., rulings, investments, production changes) unless historical news provides strong, multi-source evidence of a concrete timeline; prefer generic updates, statements, or diplomatic reactions when evidence is thin.
- Balance the tone between escalation and de-escalation: when historical news shows ongoing tensions, the true next article may be an escalation (e.g., official rebuke) rather than a compromise or solution, and vice versa for resolutions.
- When recent historical news is dominated by a single crisis narrative, explicitly consider that the next article may be completely unrelated (e.g., local politics, company-specific earnings, crime) and assign a neutral tone with no escalation.
- Avoid generating any specific future events (e.g., consortium investments, court rulings, diplomatic missions, production milestones) unless multiple historical sources provide concrete, multi-day evidence of a scheduled timeline; prefer generic statements like 'industry sources commented' or 'no new developments'.
- If the historical news shows a mature dispute with repeated stakeholder statements but no resolution, do not manufacture new diplomatic moves or operational shifts: the next article may simply be a minor update from a single company or an entirely different topic.
- When the crisis involves multiple governments and industry bodies, the true next article often narrows to a single company's isolated action (e.g., a local chip development plan, a CEO interview) rather than broad coordination or further escalation.
```
