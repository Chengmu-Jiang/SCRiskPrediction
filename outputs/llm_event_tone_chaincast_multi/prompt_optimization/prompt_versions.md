# Prompt Versions

## Version 0

- Guidance hash: `5677be43abd1`
- Forecast prompt path: `configs\llm_tone\prompts\forecast\baseline.yaml`
- Optimizer prompt path: `configs\llm_tone\prompts\optimizer\legacy.yaml`
- Val RMSE: `3.2273540083740047`
- Val MAE: `2.6020562334089123`
- Test RMSE: `2.485302745435875`
- Test MAE: `2.0674633680043546`

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

- Guidance hash: `bb9af792e295`
- Forecast prompt path: `configs\llm_tone\prompts\forecast\baseline.yaml`
- Optimizer prompt path: `configs\llm_tone\prompts\optimizer\legacy.yaml`
- Val RMSE: `3.1947579284760907`
- Val MAE: `2.469461262797819`
- Test RMSE: `2.4510484996828366`
- Test MAE: `2.0646240955271606`

### Guidance

- Avoid predicting specific court rulings, arbitration filings, or nationalization decisions unless directly confirmed by the historical trajectory; instead describe the range of possible outcomes or the ongoing uncertainty.
- Ensure the tone of the generated description aligns with the factual developments in the most recent news (e.g., diplomatic back-and-forth, mixed signals) rather than assuming a decisive resolution.
- Do not invent dates or precise sequences of events; use conditional language or describe the current state of negotiations.

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
- Avoid predicting specific court rulings, arbitration filings, or nationalization decisions unless directly confirmed by the historical trajectory; instead describe the range of possible outcomes or the ongoing uncertainty.
- Ensure the tone of the generated description aligns with the factual developments in the most recent news (e.g., diplomatic back-and-forth, mixed signals) rather than assuming a decisive resolution.
- Do not invent dates or precise sequences of events; use conditional language or describe the current state of negotiations.
```

## Version 2

- Guidance hash: `2c9ccf1de085`
- Forecast prompt path: `configs\llm_tone\prompts\forecast\baseline.yaml`
- Optimizer prompt path: `configs\llm_tone\prompts\optimizer\legacy.yaml`
- Val RMSE: `3.262981920257856`
- Val MAE: `2.6464351314679693`
- Test RMSE: `2.590215580204787`
- Test MAE: `2.1867539069123585`

### Guidance

- Avoid predicting specific court rulings, arbitration filings, or nationalization decisions unless directly confirmed by the historical trajectory; instead describe the range of possible outcomes or the ongoing uncertainty.
- Ensure the tone of the generated description aligns with the factual developments in the most recent news (e.g., diplomatic back-and-forth, mixed signals) rather than assuming a decisive resolution.
- Do not invent dates or precise sequences of events; use conditional language or describe the current state of negotiations.
- Do not anchor the entire forecast on the most recent conflict or expected decision; consider that the next effective-date news may introduce entirely new topics or shift focus to parallel developments not directly tied to the preceding narrative.
- Avoid specifying precise dates, rulings, or sequences even with conditional language unless the historical news provides a clear, direct trajectory; instead, use broad scenario language that acknowledges the potential for the story to evolve in multiple, less predictable directions.
- Ensure the tone reflects the possibility of moderate, mixed, or unrelated developments (e.g., supply chain adjustments, financial reports, geopolitical diversions) rather than assuming escalation or decisive breakthroughs in the same ongoing dispute.

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
- Avoid predicting specific court rulings, arbitration filings, or nationalization decisions unless directly confirmed by the historical trajectory; instead describe the range of possible outcomes or the ongoing uncertainty.
- Ensure the tone of the generated description aligns with the factual developments in the most recent news (e.g., diplomatic back-and-forth, mixed signals) rather than assuming a decisive resolution.
- Do not invent dates or precise sequences of events; use conditional language or describe the current state of negotiations.
- Do not anchor the entire forecast on the most recent conflict or expected decision; consider that the next effective-date news may introduce entirely new topics or shift focus to parallel developments not directly tied to the preceding narrative.
- Avoid specifying precise dates, rulings, or sequences even with conditional language unless the historical news provides a clear, direct trajectory; instead, use broad scenario language that acknowledges the potential for the story to evolve in multiple, less predictable directions.
- Ensure the tone reflects the possibility of moderate, mixed, or unrelated developments (e.g., supply chain adjustments, financial reports, geopolitical diversions) rather than assuming escalation or decisive breakthroughs in the same ongoing dispute.
```

## Version 3

- Guidance hash: `cd5c08c340df`
- Forecast prompt path: `configs\llm_tone\prompts\forecast\baseline.yaml`
- Optimizer prompt path: `configs\llm_tone\prompts\optimizer\legacy.yaml`
- Val RMSE: `3.194139536436793`
- Val MAE: `2.5581537399948475`
- Test RMSE: `2.39444672876148`
- Test MAE: `1.929239358858041`

### Guidance

- Avoid predicting specific court rulings, arbitration filings, or nationalization decisions unless directly confirmed by the historical trajectory; instead describe the range of possible outcomes or the ongoing uncertainty.
- Ensure the tone of the generated description aligns with the factual developments in the most recent news (e.g., diplomatic back-and-forth, mixed signals) rather than assuming a decisive resolution.
- Do not invent dates or precise sequences of events; use conditional language or describe the current state of negotiations.
- Do not anchor the entire forecast on the most recent conflict or expected decision; consider that the next effective-date news may introduce entirely new topics or shift focus to parallel developments not directly tied to the preceding narrative.
- Avoid specifying precise dates, rulings, or sequences even with conditional language unless the historical news provides a clear, direct trajectory; instead, use broad scenario language that acknowledges the potential for the story to evolve in multiple, less predictable directions.
- Ensure the tone reflects the possibility of moderate, mixed, or unrelated developments (e.g., supply chain adjustments, financial reports, geopolitical diversions) rather than assuming escalation or decisive breakthroughs in the same ongoing dispute.
- Do not assume that the next effective-date news will continue the dominant conflict or crisis narrative from the recent historical window; the story may pivot to an unrelated topic, a parallel development, or a moderately positive industry update.
- When the historical trajectory shows no clear next step (e.g., a hearing without ruling, a canceled meeting without substitute), describe a broad range of possible directions—including neutral, mixed, or completely new themes—rather than anchoring on escalation or prolonged deadlock.

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
- Avoid predicting specific court rulings, arbitration filings, or nationalization decisions unless directly confirmed by the historical trajectory; instead describe the range of possible outcomes or the ongoing uncertainty.
- Ensure the tone of the generated description aligns with the factual developments in the most recent news (e.g., diplomatic back-and-forth, mixed signals) rather than assuming a decisive resolution.
- Do not invent dates or precise sequences of events; use conditional language or describe the current state of negotiations.
- Do not anchor the entire forecast on the most recent conflict or expected decision; consider that the next effective-date news may introduce entirely new topics or shift focus to parallel developments not directly tied to the preceding narrative.
- Avoid specifying precise dates, rulings, or sequences even with conditional language unless the historical news provides a clear, direct trajectory; instead, use broad scenario language that acknowledges the potential for the story to evolve in multiple, less predictable directions.
- Ensure the tone reflects the possibility of moderate, mixed, or unrelated developments (e.g., supply chain adjustments, financial reports, geopolitical diversions) rather than assuming escalation or decisive breakthroughs in the same ongoing dispute.
- Do not assume that the next effective-date news will continue the dominant conflict or crisis narrative from the recent historical window; the story may pivot to an unrelated topic, a parallel development, or a moderately positive industry update.
- When the historical trajectory shows no clear next step (e.g., a hearing without ruling, a canceled meeting without substitute), describe a broad range of possible directions—including neutral, mixed, or completely new themes—rather than anchoring on escalation or prolonged deadlock.
```
