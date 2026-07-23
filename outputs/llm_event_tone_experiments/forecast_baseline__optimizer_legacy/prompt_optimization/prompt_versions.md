# Prompt Versions

## Version 0

- Guidance hash: `5677be43abd1`
- Val RMSE: `3.291866885551575`
- Val MAE: `2.585105494170652`
- Test RMSE: `3.1756506282404864`
- Test MAE: `2.6031166866107016`

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

- Guidance hash: `bfb9cd57bc10`
- Val RMSE: `3.1463959514556237`
- Val MAE: `2.5208728605065778`
- Test RMSE: `3.2819182485258502`
- Test MAE: `2.8066824691531886`

### Guidance

- Avoid inventing specific future events (dates, locations, quantified impacts) that are not strongly implied by the historical compressed news; prefer to state cautiously what the current trend suggests without fabricating details.
- Consider that the next effective-date news may be unrelated to the historical topic, lower in severity, or a continuation with reduced urgency; do not automatically assume escalation or resolution on a specific future date.
- Do not insert strong causal claims (e.g., 'due to', 'leading to', 'triggering') unless they are explicitly present in the historical news; let the tone reflect the observed uncertainty and potential for alternative outcomes.
- When historical news contains both crisis escalation and signs of diplomatic easing or supply normalization, lean toward a more neutral tone rather than predicting a sharp negative or positive event.

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
- Avoid inventing specific future events (dates, locations, quantified impacts) that are not strongly implied by the historical compressed news; prefer to state cautiously what the current trend suggests without fabricating details.
- Consider that the next effective-date news may be unrelated to the historical topic, lower in severity, or a continuation with reduced urgency; do not automatically assume escalation or resolution on a specific future date.
- Do not insert strong causal claims (e.g., 'due to', 'leading to', 'triggering') unless they are explicitly present in the historical news; let the tone reflect the observed uncertainty and potential for alternative outcomes.
- When historical news contains both crisis escalation and signs of diplomatic easing or supply normalization, lean toward a more neutral tone rather than predicting a sharp negative or positive event.
```

## Version 2

- Guidance hash: `dc4700474abd`
- Val RMSE: `3.261763347070579`
- Val MAE: `2.6046181914219777`
- Test RMSE: `3.234307950546571`
- Test MAE: `2.727861764550699`

### Guidance

- Avoid inventing specific future events (dates, locations, quantified impacts) that are not strongly implied by the historical compressed news; prefer to state cautiously what the current trend suggests without fabricating details.
- Consider that the next effective-date news may be unrelated to the historical topic, lower in severity, or a continuation with reduced urgency; do not automatically assume escalation or resolution on a specific future date.
- Do not insert strong causal claims (e.g., 'due to', 'leading to', 'triggering') unless they are explicitly present in the historical news; let the tone reflect the observed uncertainty and potential for alternative outcomes.
- When historical news contains both crisis escalation and signs of diplomatic easing or supply normalization, lean toward a more neutral tone rather than predicting a sharp negative or positive event.
- When historical news suggests a pending event (e.g., court hearing, diplomatic meeting), do not assume the outcome; instead, state the range of possibilities (e.g., ruling could escalate, ease, or remain unresolved) without inventing a specific scenario.
- Explicitly consider that the next effective-date news may be about a different topic entirely (e.g., unrelated industry development, local news, policy announcement) and calibrate tone accordingly—do not force a continuation of the crisis narrative.
- If historical news contains both conflict and de-escalation signals, avoid predicting a moderate resolution; the next news could be a renewed escalation (e.g., government condemnation, new restrictions) or a completely different angle (e.g., company strategic move).
- Do not construct future meetings, negotiations, or decisions unless the historical news specifically states they are scheduled or underway; prefer broad trend statements over detailed future events.

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
- Avoid inventing specific future events (dates, locations, quantified impacts) that are not strongly implied by the historical compressed news; prefer to state cautiously what the current trend suggests without fabricating details.
- Consider that the next effective-date news may be unrelated to the historical topic, lower in severity, or a continuation with reduced urgency; do not automatically assume escalation or resolution on a specific future date.
- Do not insert strong causal claims (e.g., 'due to', 'leading to', 'triggering') unless they are explicitly present in the historical news; let the tone reflect the observed uncertainty and potential for alternative outcomes.
- When historical news contains both crisis escalation and signs of diplomatic easing or supply normalization, lean toward a more neutral tone rather than predicting a sharp negative or positive event.
- When historical news suggests a pending event (e.g., court hearing, diplomatic meeting), do not assume the outcome; instead, state the range of possibilities (e.g., ruling could escalate, ease, or remain unresolved) without inventing a specific scenario.
- Explicitly consider that the next effective-date news may be about a different topic entirely (e.g., unrelated industry development, local news, policy announcement) and calibrate tone accordingly—do not force a continuation of the crisis narrative.
- If historical news contains both conflict and de-escalation signals, avoid predicting a moderate resolution; the next news could be a renewed escalation (e.g., government condemnation, new restrictions) or a completely different angle (e.g., company strategic move).
- Do not construct future meetings, negotiations, or decisions unless the historical news specifically states they are scheduled or underway; prefer broad trend statements over detailed future events.
```

## Version 3

- Guidance hash: `9c783ce4aa64`
- Val RMSE: `3.368577847126004`
- Val MAE: `2.7293355682552445`
- Test RMSE: `3.4952508302010408`
- Test MAE: `3.039236929817471`

### Guidance

- Avoid inventing specific future events (dates, locations, quantified impacts) that are not strongly implied by the historical compressed news; prefer to state cautiously what the current trend suggests without fabricating details.
- Consider that the next effective-date news may be unrelated to the historical topic, lower in severity, or a continuation with reduced urgency; do not automatically assume escalation or resolution on a specific future date.
- Do not insert strong causal claims (e.g., 'due to', 'leading to', 'triggering') unless they are explicitly present in the historical news; let the tone reflect the observed uncertainty and potential for alternative outcomes.
- When historical news contains both crisis escalation and signs of diplomatic easing or supply normalization, lean toward a more neutral tone rather than predicting a sharp negative or positive event.
- When historical news suggests a pending event (e.g., court hearing, diplomatic meeting), do not assume the outcome; instead, state the range of possibilities (e.g., ruling could escalate, ease, or remain unresolved) without inventing a specific scenario.
- Explicitly consider that the next effective-date news may be about a different topic entirely (e.g., unrelated industry development, local news, policy announcement) and calibrate tone accordingly—do not force a continuation of the crisis narrative.
- If historical news contains both conflict and de-escalation signals, avoid predicting a moderate resolution; the next news could be a renewed escalation (e.g., government condemnation, new restrictions) or a completely different angle (e.g., company strategic move).
- Do not construct future meetings, negotiations, or decisions unless the historical news specifically states they are scheduled or underway; prefer broad trend statements over detailed future events.
- Explicitly state that the next effective-date news may be about a completely different topic (e.g., unrelated industry development, local news, policy announcement) and if so, the tone should reflect the default severity of that new topic—often neutral or unrelated—rather than forcing a continuation of the historical crisis narrative.
- When historical news contains both escalation and de-escalation signals, do not assume a moderate continuation; the next news could be a renewed escalation (e.g., new restrictions, government condemnation) or a completely different angle (e.g., company strategic move). Avoid constructing specific future events like court rulings, meeting outcomes, or diplomatic deals unless their occurrence and predictable direction are strongly implied by the historical news; instead, state the range of possibilities without committing to one scenario.

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
- Avoid inventing specific future events (dates, locations, quantified impacts) that are not strongly implied by the historical compressed news; prefer to state cautiously what the current trend suggests without fabricating details.
- Consider that the next effective-date news may be unrelated to the historical topic, lower in severity, or a continuation with reduced urgency; do not automatically assume escalation or resolution on a specific future date.
- Do not insert strong causal claims (e.g., 'due to', 'leading to', 'triggering') unless they are explicitly present in the historical news; let the tone reflect the observed uncertainty and potential for alternative outcomes.
- When historical news contains both crisis escalation and signs of diplomatic easing or supply normalization, lean toward a more neutral tone rather than predicting a sharp negative or positive event.
- When historical news suggests a pending event (e.g., court hearing, diplomatic meeting), do not assume the outcome; instead, state the range of possibilities (e.g., ruling could escalate, ease, or remain unresolved) without inventing a specific scenario.
- Explicitly consider that the next effective-date news may be about a different topic entirely (e.g., unrelated industry development, local news, policy announcement) and calibrate tone accordingly—do not force a continuation of the crisis narrative.
- If historical news contains both conflict and de-escalation signals, avoid predicting a moderate resolution; the next news could be a renewed escalation (e.g., government condemnation, new restrictions) or a completely different angle (e.g., company strategic move).
- Do not construct future meetings, negotiations, or decisions unless the historical news specifically states they are scheduled or underway; prefer broad trend statements over detailed future events.
- Explicitly state that the next effective-date news may be about a completely different topic (e.g., unrelated industry development, local news, policy announcement) and if so, the tone should reflect the default severity of that new topic—often neutral or unrelated—rather than forcing a continuation of the historical crisis narrative.
- When historical news contains both escalation and de-escalation signals, do not assume a moderate continuation; the next news could be a renewed escalation (e.g., new restrictions, government condemnation) or a completely different angle (e.g., company strategic move). Avoid constructing specific future events like court rulings, meeting outcomes, or diplomatic deals unless their occurrence and predictable direction are strongly implied by the historical news; instead, state the range of possibilities without committing to one scenario.
```
