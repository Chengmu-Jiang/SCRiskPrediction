# Prompt Versions

## Version 0

- Guidance hash: `5677be43abd1`
- Forecast prompt path: `configs\llm_tone\prompts\forecast\baseline_one_shot_v1.yaml`
- Optimizer prompt path: `configs\llm_tone\prompts\optimizer\anti_overfit_v1.yaml`
- Val RMSE: `3.39385606175654`
- Val MAE: `2.6952246719431208`
- Test RMSE: `3.3596079679087594`
- Test MAE: `2.955350222641415`

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

- Guidance hash: `d157a506dc95`
- Forecast prompt path: `configs\llm_tone\prompts\forecast\baseline_one_shot_v1.yaml`
- Optimizer prompt path: `configs\llm_tone\prompts\optimizer\anti_overfit_v1.yaml`
- Val RMSE: `3.3246858910587562`
- Val MAE: `2.6853213776483686`
- Test RMSE: `3.103875079724276`
- Test MAE: `2.5626869910180163`

### Guidance

- When forecasting legal or regulatory outcomes, avoid predicting final decisions (e.g., court orders, seizure revocations) unless a binding deadline is explicitly set in historical news; instead describe hearings, appeals, or pending deliberations.
- For geopolitical disputes, prioritize forecasting government-level communications (statements, calls, cancellations) over corporate investment or production changes, as diplomatic signals often precede business actions.
- Limit the scope of future-event predictions to the most immediate scheduled events (e.g., hearings, visits) or known policy steps; do not invent new consortiums, investment rounds, or capacity expansions unless they are directly announced in the historical timeline.

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
- When forecasting legal or regulatory outcomes, avoid predicting final decisions (e.g., court orders, seizure revocations) unless a binding deadline is explicitly set in historical news; instead describe hearings, appeals, or pending deliberations.
- For geopolitical disputes, prioritize forecasting government-level communications (statements, calls, cancellations) over corporate investment or production changes, as diplomatic signals often precede business actions.
- Limit the scope of future-event predictions to the most immediate scheduled events (e.g., hearings, visits) or known policy steps; do not invent new consortiums, investment rounds, or capacity expansions unless they are directly announced in the historical timeline.
```

## Version 2

- Guidance hash: `47889266b876`
- Forecast prompt path: `configs\llm_tone\prompts\forecast\baseline_one_shot_v1.yaml`
- Optimizer prompt path: `configs\llm_tone\prompts\optimizer\anti_overfit_v1.yaml`
- Val RMSE: `3.1964799598161386`
- Val MAE: `2.487840686434637`
- Test RMSE: `3.012970312593983`
- Test MAE: `2.4702326339742466`

### Guidance

- When forecasting legal or regulatory outcomes, avoid predicting final decisions (e.g., court orders, seizure revocations) unless a binding deadline is explicitly set in historical news; instead describe hearings, appeals, or pending deliberations.
- For geopolitical disputes, prioritize forecasting government-level communications (statements, calls, cancellations) over corporate investment or production changes, as diplomatic signals often precede business actions.
- Limit the scope of future-event predictions to the most immediate scheduled events (e.g., hearings, visits) or known policy steps; do not invent new consortiums, investment rounds, or capacity expansions unless they are directly announced in the historical timeline.
- Avoid forecasting multi-party legal proceedings (hearings, appeals, parliamentary debates) or industry roundtables unless a specific date or binding deadline is explicitly stated in the historical timeline; the next event is often a single government/company statement, not a convened meeting.
- Do not predict production halts, plant idle periods, or worsening supply shortages unless the historical news directly describes such an action as imminent or scheduled; many high-loss examples show the model inventing shutdowns when the actual next event is a mitigating corporate or diplomatic move.
- When the historical timeline contains diplomatic communications (statements, visa/canceled visits, formal demands), prioritize these as the most likely next effective-date news over hypothetical business responses or legal appeals, as government signals frequently precede corporate reactions.

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
- When forecasting legal or regulatory outcomes, avoid predicting final decisions (e.g., court orders, seizure revocations) unless a binding deadline is explicitly set in historical news; instead describe hearings, appeals, or pending deliberations.
- For geopolitical disputes, prioritize forecasting government-level communications (statements, calls, cancellations) over corporate investment or production changes, as diplomatic signals often precede business actions.
- Limit the scope of future-event predictions to the most immediate scheduled events (e.g., hearings, visits) or known policy steps; do not invent new consortiums, investment rounds, or capacity expansions unless they are directly announced in the historical timeline.
- Avoid forecasting multi-party legal proceedings (hearings, appeals, parliamentary debates) or industry roundtables unless a specific date or binding deadline is explicitly stated in the historical timeline; the next event is often a single government/company statement, not a convened meeting.
- Do not predict production halts, plant idle periods, or worsening supply shortages unless the historical news directly describes such an action as imminent or scheduled; many high-loss examples show the model inventing shutdowns when the actual next event is a mitigating corporate or diplomatic move.
- When the historical timeline contains diplomatic communications (statements, visa/canceled visits, formal demands), prioritize these as the most likely next effective-date news over hypothetical business responses or legal appeals, as government signals frequently precede corporate reactions.
```

## Version 3

- Guidance hash: `e0731301b929`
- Forecast prompt path: `configs\llm_tone\prompts\forecast\baseline_one_shot_v1.yaml`
- Optimizer prompt path: `configs\llm_tone\prompts\optimizer\anti_overfit_v1.yaml`
- Val RMSE: `3.375909024856822`
- Val MAE: `2.709244733626836`
- Test RMSE: `3.201694193411936`
- Test MAE: `2.7339376297932745`

### Guidance

- When forecasting legal or regulatory outcomes, avoid predicting final decisions (e.g., court orders, seizure revocations) unless a binding deadline is explicitly set in historical news; instead describe hearings, appeals, or pending deliberations.
- For geopolitical disputes, prioritize forecasting government-level communications (statements, calls, cancellations) over corporate investment or production changes, as diplomatic signals often precede business actions.
- Limit the scope of future-event predictions to the most immediate scheduled events (e.g., hearings, visits) or known policy steps; do not invent new consortiums, investment rounds, or capacity expansions unless they are directly announced in the historical timeline.
- Avoid forecasting multi-party legal proceedings (hearings, appeals, parliamentary debates) or industry roundtables unless a specific date or binding deadline is explicitly stated in the historical timeline; the next event is often a single government/company statement, not a convened meeting.
- Do not predict production halts, plant idle periods, or worsening supply shortages unless the historical news directly describes such an action as imminent or scheduled; many high-loss examples show the model inventing shutdowns when the actual next event is a mitigating corporate or diplomatic move.
- When the historical timeline contains diplomatic communications (statements, visa/canceled visits, formal demands), prioritize these as the most likely next effective-date news over hypothetical business responses or legal appeals, as government signals frequently precede corporate reactions.
- When the most recent historical news explicitly describes a specific action (e.g., a minister cancellation, a corporate announcement, a government statement), treat that as the next likely event; do not forecast a follow-up response or convened meeting unless a date or binding deadline is attached in the historical text.
- In supply-chain or dispute narratives, avoid predicting formal responses (e.g., court rulings, government rejections, custodians' replies) unless the historical timeline provides explicit indication of such a step or deadline; without that, the next event is typically another statement or action from the same actor.
- Do not embed legal processes (WTO filings, hearing dates, parliamentary debates) into forecasts unless the historical news explicitly mentions those steps or scheduled dates; otherwise the next event is a simpler diplomatic statement or corporate action already hinted in the timeline.

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
- When forecasting legal or regulatory outcomes, avoid predicting final decisions (e.g., court orders, seizure revocations) unless a binding deadline is explicitly set in historical news; instead describe hearings, appeals, or pending deliberations.
- For geopolitical disputes, prioritize forecasting government-level communications (statements, calls, cancellations) over corporate investment or production changes, as diplomatic signals often precede business actions.
- Limit the scope of future-event predictions to the most immediate scheduled events (e.g., hearings, visits) or known policy steps; do not invent new consortiums, investment rounds, or capacity expansions unless they are directly announced in the historical timeline.
- Avoid forecasting multi-party legal proceedings (hearings, appeals, parliamentary debates) or industry roundtables unless a specific date or binding deadline is explicitly stated in the historical timeline; the next event is often a single government/company statement, not a convened meeting.
- Do not predict production halts, plant idle periods, or worsening supply shortages unless the historical news directly describes such an action as imminent or scheduled; many high-loss examples show the model inventing shutdowns when the actual next event is a mitigating corporate or diplomatic move.
- When the historical timeline contains diplomatic communications (statements, visa/canceled visits, formal demands), prioritize these as the most likely next effective-date news over hypothetical business responses or legal appeals, as government signals frequently precede corporate reactions.
- When the most recent historical news explicitly describes a specific action (e.g., a minister cancellation, a corporate announcement, a government statement), treat that as the next likely event; do not forecast a follow-up response or convened meeting unless a date or binding deadline is attached in the historical text.
- In supply-chain or dispute narratives, avoid predicting formal responses (e.g., court rulings, government rejections, custodians' replies) unless the historical timeline provides explicit indication of such a step or deadline; without that, the next event is typically another statement or action from the same actor.
- Do not embed legal processes (WTO filings, hearing dates, parliamentary debates) into forecasts unless the historical news explicitly mentions those steps or scheduled dates; otherwise the next event is a simpler diplomatic statement or corporate action already hinted in the timeline.
```
