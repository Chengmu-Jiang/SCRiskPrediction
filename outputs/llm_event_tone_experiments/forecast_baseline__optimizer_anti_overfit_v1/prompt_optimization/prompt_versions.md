# Prompt Versions

## Version 0

- Guidance hash: `5677be43abd1`
- Forecast prompt path: `configs\llm_tone\prompts\forecast\baseline.yaml`
- Optimizer prompt path: `configs\llm_tone\prompts\optimizer\anti_overfit_v1.yaml`
- Val RMSE: `3.224837465682222`
- Val MAE: `2.530897176185632`
- Test RMSE: `3.157065880335534`
- Test MAE: `2.6577045429116346`

### Guidance

(original prompt; no extra guidance)

### Prompt Template

```text
You are forecasting the next effective news event for a product/topic.

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
- Use only the historical sequence as evidence.
- Focus on the likely event narrative and tone drivers.
- Do not output ArticleScore, AvgTone, labels, bullet points, or explanations.
- Do not include hidden reasoning, chain-of-thought, analysis notes, or <think> tags.
- Use 80 to 180 words.
- Return only the generated future-event description.
```

## Version 1

- Guidance hash: `27b9ba9537da`
- Forecast prompt path: `configs\llm_tone\prompts\forecast\baseline.yaml`
- Optimizer prompt path: `configs\llm_tone\prompts\optimizer\anti_overfit_v1.yaml`
- Val RMSE: `3.5916808103915714`
- Val MAE: `2.8541363195416256`
- Test RMSE: `3.492783431410704`
- Test MAE: `3.022455868341934`

### Guidance

- Prefer forecasting near-term newsworthy statements, meetings, or incremental policy actions over definitive legal rulings or operational shutdowns unless the historical context strongly supports an imminent decision.
- In supply-chain and geopolitical disputes, calibrate impact severity to actual signals from official sources (ministry statements, company announcements) rather than extrapolating worst-case or best-case supply disruptions.
- Avoid inventing specific dates for future events unless the historical news contains a clear scheduled date or deadline; use relative time references or ‘expected within [timeframe]’ when the timeline is uncertain.

### Prompt Template

```text
You are forecasting the next effective news event for a product/topic.

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
- Use only the historical sequence as evidence.
- Focus on the likely event narrative and tone drivers.
- Do not output ArticleScore, AvgTone, labels, bullet points, or explanations.
- Do not include hidden reasoning, chain-of-thought, analysis notes, or <think> tags.
- Use 80 to 180 words.
- Return only the generated future-event description.

Additional tone-calibration guidance learned from training feedback:
- Prefer forecasting near-term newsworthy statements, meetings, or incremental policy actions over definitive legal rulings or operational shutdowns unless the historical context strongly supports an imminent decision.
- In supply-chain and geopolitical disputes, calibrate impact severity to actual signals from official sources (ministry statements, company announcements) rather than extrapolating worst-case or best-case supply disruptions.
- Avoid inventing specific dates for future events unless the historical news contains a clear scheduled date or deadline; use relative time references or ‘expected within [timeframe]’ when the timeline is uncertain.
```

## Version 2

- Guidance hash: `79ed0140eb8b`
- Forecast prompt path: `configs\llm_tone\prompts\forecast\baseline.yaml`
- Optimizer prompt path: `configs\llm_tone\prompts\optimizer\anti_overfit_v1.yaml`
- Val RMSE: `3.324142778949361`
- Val MAE: `2.7262989703194043`
- Test RMSE: `3.1488544447963505`
- Test MAE: `2.6882397730578362`

### Guidance

- Prefer forecasting near-term newsworthy statements, meetings, or incremental policy actions over definitive legal rulings or operational shutdowns unless the historical context strongly supports an imminent decision.
- In supply-chain and geopolitical disputes, calibrate impact severity to actual signals from official sources (ministry statements, company announcements) rather than extrapolating worst-case or best-case supply disruptions.
- Avoid inventing specific dates for future events unless the historical news contains a clear scheduled date or deadline; use relative time references or ‘expected within [timeframe]’ when the timeline is uncertain.
- When the historical news contains sustained tension, competing accusations, or canceled meetings, prefer forecasting an escalation or continued stalemate over a negotiated resolution, unless a clear de-escalation signal appears.
- Stay within the set of actors and actions explicitly mentioned in recent historical news; do not introduce new companies, agreements, or events that are not directly foreshadowed by the source material.
- Avoid assigning a precise date (e.g., 'on December 9') unless the historical news contains a scheduled event or clear deadline; use relative timeframes like 'within weeks' or 'expected in early 2026' when the timeline is uncertain.

### Prompt Template

```text
You are forecasting the next effective news event for a product/topic.

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
- Use only the historical sequence as evidence.
- Focus on the likely event narrative and tone drivers.
- Do not output ArticleScore, AvgTone, labels, bullet points, or explanations.
- Do not include hidden reasoning, chain-of-thought, analysis notes, or <think> tags.
- Use 80 to 180 words.
- Return only the generated future-event description.

Additional tone-calibration guidance learned from training feedback:
- Prefer forecasting near-term newsworthy statements, meetings, or incremental policy actions over definitive legal rulings or operational shutdowns unless the historical context strongly supports an imminent decision.
- In supply-chain and geopolitical disputes, calibrate impact severity to actual signals from official sources (ministry statements, company announcements) rather than extrapolating worst-case or best-case supply disruptions.
- Avoid inventing specific dates for future events unless the historical news contains a clear scheduled date or deadline; use relative time references or ‘expected within [timeframe]’ when the timeline is uncertain.
- When the historical news contains sustained tension, competing accusations, or canceled meetings, prefer forecasting an escalation or continued stalemate over a negotiated resolution, unless a clear de-escalation signal appears.
- Stay within the set of actors and actions explicitly mentioned in recent historical news; do not introduce new companies, agreements, or events that are not directly foreshadowed by the source material.
- Avoid assigning a precise date (e.g., 'on December 9') unless the historical news contains a scheduled event or clear deadline; use relative timeframes like 'within weeks' or 'expected in early 2026' when the timeline is uncertain.
```

## Version 3

- Guidance hash: `e42681b6cd39`
- Forecast prompt path: `configs\llm_tone\prompts\forecast\baseline.yaml`
- Optimizer prompt path: `configs\llm_tone\prompts\optimizer\anti_overfit_v1.yaml`
- Val RMSE: `3.230328772934058`
- Val MAE: `2.6623304949337374`
- Test RMSE: `3.378324094183675`
- Test MAE: `2.8960706897940094`

### Guidance

- Prefer forecasting near-term newsworthy statements, meetings, or incremental policy actions over definitive legal rulings or operational shutdowns unless the historical context strongly supports an imminent decision.
- In supply-chain and geopolitical disputes, calibrate impact severity to actual signals from official sources (ministry statements, company announcements) rather than extrapolating worst-case or best-case supply disruptions.
- Avoid inventing specific dates for future events unless the historical news contains a clear scheduled date or deadline; use relative time references or ‘expected within [timeframe]’ when the timeline is uncertain.
- When the historical news contains sustained tension, competing accusations, or canceled meetings, prefer forecasting an escalation or continued stalemate over a negotiated resolution, unless a clear de-escalation signal appears.
- Stay within the set of actors and actions explicitly mentioned in recent historical news; do not introduce new companies, agreements, or events that are not directly foreshadowed by the source material.
- Avoid assigning a precise date (e.g., 'on December 9') unless the historical news contains a scheduled event or clear deadline; use relative timeframes like 'within weeks' or 'expected in early 2026' when the timeline is uncertain.
- When the historical news contains ongoing legal, diplomatic, or corporate disputes with no imminent decision event, avoid forecasting the outcome of a court case or the launch of formal international proceedings; instead forecast the next observable step that has been explicitly mentioned or is the typical near-term action (e.g., a party issues a statement, a regulator sets a deadline, a delegation meets).
- If the historical record shows only accusations, canceled meetings, or supply disruptions without a clear de-escalation signal, prefer forecasting an adaptive response (e.g., the affected company secures alternative sources, an industry body issues a warning, or a government makes a non-binding request) over an escalation in conflict or a negotiated settlement, because the ground truth often reflects operational adjustment rather than a decisive change in the dispute.
- When supply constraints or geopolitical tensions are described as ongoing in the source material, anchor the forecast on supply-chain or policy consequences that have already been named (e.g., temporary permits, production adjustments, or official statements) and avoid inferring new trade measures, production shutdowns, or industry-wide collaborations that are not directly foreshadowed by a named actor or scheduled event.

### Prompt Template

```text
You are forecasting the next effective news event for a product/topic.

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
- Use only the historical sequence as evidence.
- Focus on the likely event narrative and tone drivers.
- Do not output ArticleScore, AvgTone, labels, bullet points, or explanations.
- Do not include hidden reasoning, chain-of-thought, analysis notes, or <think> tags.
- Use 80 to 180 words.
- Return only the generated future-event description.

Additional tone-calibration guidance learned from training feedback:
- Prefer forecasting near-term newsworthy statements, meetings, or incremental policy actions over definitive legal rulings or operational shutdowns unless the historical context strongly supports an imminent decision.
- In supply-chain and geopolitical disputes, calibrate impact severity to actual signals from official sources (ministry statements, company announcements) rather than extrapolating worst-case or best-case supply disruptions.
- Avoid inventing specific dates for future events unless the historical news contains a clear scheduled date or deadline; use relative time references or ‘expected within [timeframe]’ when the timeline is uncertain.
- When the historical news contains sustained tension, competing accusations, or canceled meetings, prefer forecasting an escalation or continued stalemate over a negotiated resolution, unless a clear de-escalation signal appears.
- Stay within the set of actors and actions explicitly mentioned in recent historical news; do not introduce new companies, agreements, or events that are not directly foreshadowed by the source material.
- Avoid assigning a precise date (e.g., 'on December 9') unless the historical news contains a scheduled event or clear deadline; use relative timeframes like 'within weeks' or 'expected in early 2026' when the timeline is uncertain.
- When the historical news contains ongoing legal, diplomatic, or corporate disputes with no imminent decision event, avoid forecasting the outcome of a court case or the launch of formal international proceedings; instead forecast the next observable step that has been explicitly mentioned or is the typical near-term action (e.g., a party issues a statement, a regulator sets a deadline, a delegation meets).
- If the historical record shows only accusations, canceled meetings, or supply disruptions without a clear de-escalation signal, prefer forecasting an adaptive response (e.g., the affected company secures alternative sources, an industry body issues a warning, or a government makes a non-binding request) over an escalation in conflict or a negotiated settlement, because the ground truth often reflects operational adjustment rather than a decisive change in the dispute.
- When supply constraints or geopolitical tensions are described as ongoing in the source material, anchor the forecast on supply-chain or policy consequences that have already been named (e.g., temporary permits, production adjustments, or official statements) and avoid inferring new trade measures, production shutdowns, or industry-wide collaborations that are not directly foreshadowed by a named actor or scheduled event.
```
