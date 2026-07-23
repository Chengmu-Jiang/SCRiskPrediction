# Prompt Versions

## Version 0

- Guidance hash: `5677be43abd1`
- Forecast prompt path: `configs\llm_tone\prompts\forecast\baseline.yaml`
- Optimizer prompt path: `configs\llm_tone\prompts\optimizer\legacy.yaml`
- Val RMSE: `3.1484080647930557`
- Val MAE: `2.545589398154099`
- Test RMSE: `2.5835779862873958`
- Test MAE: `2.1211927572306535`

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

- Guidance hash: `ffece74cf95a`
- Forecast prompt path: `configs\llm_tone\prompts\forecast\baseline.yaml`
- Optimizer prompt path: `configs\llm_tone\prompts\optimizer\legacy.yaml`
- Val RMSE: `3.1992245241743023`
- Val MAE: `2.6295982546705834`
- Test RMSE: `2.4052196103019816`
- Test MAE: `2.06190786842397`

### Guidance

- Do not invent specific dates, production halts, or legal rulings unless strongly signaled by historical news; when historical news shows diplomatic talks or uncertainty, keep predictions ambiguous or aligned with observable signals.
- When historical news is ambiguous or shows de-escalation attempts, favor neutral or constructive tones (e.g., talks ongoing, parties seeking resolution) over escalation or crisis.
- If historical news is dominated by one actor or narrative, consider that the true next article may come from a different perspective (e.g., Chinese government statement) or even be off-topic; avoid assuming continuation of the most dramatic narrative.
- For topics with geopolitical tension, do not automatically predict escalation; the true news may reflect a softer stance or a call for dialogue.

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
- Do not invent specific dates, production halts, or legal rulings unless strongly signaled by historical news; when historical news shows diplomatic talks or uncertainty, keep predictions ambiguous or aligned with observable signals.
- When historical news is ambiguous or shows de-escalation attempts, favor neutral or constructive tones (e.g., talks ongoing, parties seeking resolution) over escalation or crisis.
- If historical news is dominated by one actor or narrative, consider that the true next article may come from a different perspective (e.g., Chinese government statement) or even be off-topic; avoid assuming continuation of the most dramatic narrative.
- For topics with geopolitical tension, do not automatically predict escalation; the true news may reflect a softer stance or a call for dialogue.
```

## Version 2

- Guidance hash: `263db76849d5`
- Forecast prompt path: `configs\llm_tone\prompts\forecast\baseline.yaml`
- Optimizer prompt path: `configs\llm_tone\prompts\optimizer\legacy.yaml`
- Val RMSE: `3.268103491765822`
- Val MAE: `2.6830483430426617`
- Test RMSE: `2.686758715681168`
- Test MAE: `2.2213358403622054`

### Guidance

- Do not invent specific dates, production halts, or legal rulings unless strongly signaled by historical news; when historical news shows diplomatic talks or uncertainty, keep predictions ambiguous or aligned with observable signals.
- When historical news is ambiguous or shows de-escalation attempts, favor neutral or constructive tones (e.g., talks ongoing, parties seeking resolution) over escalation or crisis.
- If historical news is dominated by one actor or narrative, consider that the true next article may come from a different perspective (e.g., Chinese government statement) or even be off-topic; avoid assuming continuation of the most dramatic narrative.
- For topics with geopolitical tension, do not automatically predict escalation; the true news may reflect a softer stance or a call for dialogue.
- When historical news shows escalating conflict, canceled visits, or sharp criticism, do not automatically predict de-escalation or constructive talks; instead, reflect the likely continuation or intensification of the confrontational tone.
- Explicitly consider that the next article may be off-topic entirely, especially when the historical news compresses many entries on a single narrow event; assign higher probability to a shift in subject matter rather than assuming a linear narrative continuation.
- Do not invent specific meetings, statements, or diplomatic steps unless the historical news provides strong and direct evidence that such events are scheduled or have occurred; prefer generic descriptions of the situation when details are absent.

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
- Do not invent specific dates, production halts, or legal rulings unless strongly signaled by historical news; when historical news shows diplomatic talks or uncertainty, keep predictions ambiguous or aligned with observable signals.
- When historical news is ambiguous or shows de-escalation attempts, favor neutral or constructive tones (e.g., talks ongoing, parties seeking resolution) over escalation or crisis.
- If historical news is dominated by one actor or narrative, consider that the true next article may come from a different perspective (e.g., Chinese government statement) or even be off-topic; avoid assuming continuation of the most dramatic narrative.
- For topics with geopolitical tension, do not automatically predict escalation; the true news may reflect a softer stance or a call for dialogue.
- When historical news shows escalating conflict, canceled visits, or sharp criticism, do not automatically predict de-escalation or constructive talks; instead, reflect the likely continuation or intensification of the confrontational tone.
- Explicitly consider that the next article may be off-topic entirely, especially when the historical news compresses many entries on a single narrow event; assign higher probability to a shift in subject matter rather than assuming a linear narrative continuation.
- Do not invent specific meetings, statements, or diplomatic steps unless the historical news provides strong and direct evidence that such events are scheduled or have occurred; prefer generic descriptions of the situation when details are absent.
```

## Version 3

- Guidance hash: `e2aaf539fa25`
- Forecast prompt path: `configs\llm_tone\prompts\forecast\baseline.yaml`
- Optimizer prompt path: `configs\llm_tone\prompts\optimizer\legacy.yaml`
- Val RMSE: `3.35781769318566`
- Val MAE: `2.7302225061789`
- Test RMSE: `2.64540163231434`
- Test MAE: `2.2847832515931565`

### Guidance

- Do not invent specific dates, production halts, or legal rulings unless strongly signaled by historical news; when historical news shows diplomatic talks or uncertainty, keep predictions ambiguous or aligned with observable signals.
- When historical news is ambiguous or shows de-escalation attempts, favor neutral or constructive tones (e.g., talks ongoing, parties seeking resolution) over escalation or crisis.
- If historical news is dominated by one actor or narrative, consider that the true next article may come from a different perspective (e.g., Chinese government statement) or even be off-topic; avoid assuming continuation of the most dramatic narrative.
- For topics with geopolitical tension, do not automatically predict escalation; the true news may reflect a softer stance or a call for dialogue.
- When historical news shows escalating conflict, canceled visits, or sharp criticism, do not automatically predict de-escalation or constructive talks; instead, reflect the likely continuation or intensification of the confrontational tone.
- Explicitly consider that the next article may be off-topic entirely, especially when the historical news compresses many entries on a single narrow event; assign higher probability to a shift in subject matter rather than assuming a linear narrative continuation.
- Do not invent specific meetings, statements, or diplomatic steps unless the historical news provides strong and direct evidence that such events are scheduled or have occurred; prefer generic descriptions of the situation when details are absent.
- When historical news compresses many entries on a single narrow event (e.g., Nexperia standoff, automaker chip crisis), explicitly assign high probability to the next article being completely off-topic—an unrelated local news story, a different industry announcement, or a human-interest piece—rather than continuing the same narrative.
- If the historical news contains any off-topic articles interspersed (e.g., school hiring, soldier mourning, lodge legacy), treat those as strong signals that the sequence is not purely thematic; predict tone-neutral or constructive descriptions for the next article if it returns to the main topic, but default to expecting a different subject entirely.

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
- Do not invent specific dates, production halts, or legal rulings unless strongly signaled by historical news; when historical news shows diplomatic talks or uncertainty, keep predictions ambiguous or aligned with observable signals.
- When historical news is ambiguous or shows de-escalation attempts, favor neutral or constructive tones (e.g., talks ongoing, parties seeking resolution) over escalation or crisis.
- If historical news is dominated by one actor or narrative, consider that the true next article may come from a different perspective (e.g., Chinese government statement) or even be off-topic; avoid assuming continuation of the most dramatic narrative.
- For topics with geopolitical tension, do not automatically predict escalation; the true news may reflect a softer stance or a call for dialogue.
- When historical news shows escalating conflict, canceled visits, or sharp criticism, do not automatically predict de-escalation or constructive talks; instead, reflect the likely continuation or intensification of the confrontational tone.
- Explicitly consider that the next article may be off-topic entirely, especially when the historical news compresses many entries on a single narrow event; assign higher probability to a shift in subject matter rather than assuming a linear narrative continuation.
- Do not invent specific meetings, statements, or diplomatic steps unless the historical news provides strong and direct evidence that such events are scheduled or have occurred; prefer generic descriptions of the situation when details are absent.
- When historical news compresses many entries on a single narrow event (e.g., Nexperia standoff, automaker chip crisis), explicitly assign high probability to the next article being completely off-topic—an unrelated local news story, a different industry announcement, or a human-interest piece—rather than continuing the same narrative.
- If the historical news contains any off-topic articles interspersed (e.g., school hiring, soldier mourning, lodge legacy), treat those as strong signals that the sequence is not purely thematic; predict tone-neutral or constructive descriptions for the next article if it returns to the main topic, but default to expecting a different subject entirely.
```
