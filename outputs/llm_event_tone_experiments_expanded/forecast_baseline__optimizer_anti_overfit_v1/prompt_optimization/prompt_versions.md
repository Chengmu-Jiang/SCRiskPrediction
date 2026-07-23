# Prompt Versions

## Version 0

- Guidance hash: `5677be43abd1`
- Forecast prompt path: `configs\llm_tone\prompts\forecast\baseline.yaml`
- Optimizer prompt path: `configs\llm_tone\prompts\optimizer\anti_overfit_v1.yaml`
- Val RMSE: `3.168648935813816`
- Val MAE: `2.5169073667205066`
- Test RMSE: `2.62668387549759`
- Test MAE: `2.1262388311326497`

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

- Guidance hash: `776ece4567be`
- Forecast prompt path: `configs\llm_tone\prompts\forecast\baseline.yaml`
- Optimizer prompt path: `configs\llm_tone\prompts\optimizer\anti_overfit_v1.yaml`
- Val RMSE: `3.1865357685502538`
- Val MAE: `2.5184407413879724`
- Test RMSE: `2.3662980216058878`
- Test MAE: `1.9490167225124275`

### Guidance

- Do not predict decisive legal or regulatory outcomes (e.g., court rulings, arbitration filings, nationalization) unless the most recent historical news explicitly indicates such a decision is imminent; instead, calibrate forecasts to ongoing diplomatic exchanges, industry responses, or policy statements that reflect the current tension level.
- For automaker supply disruptions, avoid assuming specific production halts or shutdowns unless directly confirmed by recent history; true news may cover strategic adaptations (e.g., chip development, financial results, supply resumption) that shift focus away from crisis escalation.
- Ensure event specificity aligns with available signals: do not invent detailed events (e.g., temporary compromise deals, new export bans, parliamentary debates) that lack clear foreshadowing in the preceding compressed news; prefer describing likely next steps for actors already mentioned, such as further talks, operational pivots, or official statements.

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
- Do not predict decisive legal or regulatory outcomes (e.g., court rulings, arbitration filings, nationalization) unless the most recent historical news explicitly indicates such a decision is imminent; instead, calibrate forecasts to ongoing diplomatic exchanges, industry responses, or policy statements that reflect the current tension level.
- For automaker supply disruptions, avoid assuming specific production halts or shutdowns unless directly confirmed by recent history; true news may cover strategic adaptations (e.g., chip development, financial results, supply resumption) that shift focus away from crisis escalation.
- Ensure event specificity aligns with available signals: do not invent detailed events (e.g., temporary compromise deals, new export bans, parliamentary debates) that lack clear foreshadowing in the preceding compressed news; prefer describing likely next steps for actors already mentioned, such as further talks, operational pivots, or official statements.
```

## Version 2

- Guidance hash: `881893935454`
- Forecast prompt path: `configs\llm_tone\prompts\forecast\baseline.yaml`
- Optimizer prompt path: `configs\llm_tone\prompts\optimizer\anti_overfit_v1.yaml`
- Val RMSE: `3.172519877539279`
- Val MAE: `2.580919792875857`
- Test RMSE: `2.950366426852521`
- Test MAE: `2.3530864661398274`

### Guidance

- Do not predict decisive legal or regulatory outcomes (e.g., court rulings, arbitration filings, nationalization) unless the most recent historical news explicitly indicates such a decision is imminent; instead, calibrate forecasts to ongoing diplomatic exchanges, industry responses, or policy statements that reflect the current tension level.
- For automaker supply disruptions, avoid assuming specific production halts or shutdowns unless directly confirmed by recent history; true news may cover strategic adaptations (e.g., chip development, financial results, supply resumption) that shift focus away from crisis escalation.
- Ensure event specificity aligns with available signals: do not invent detailed events (e.g., temporary compromise deals, new export bans, parliamentary debates) that lack clear foreshadowing in the preceding compressed news; prefer describing likely next steps for actors already mentioned, such as further talks, operational pivots, or official statements.
- Prefer forecasting financial results, strategic announcements, or industry operational shifts (e.g., chip development, profit warnings) over diplomatic or legal breakthroughs unless the latest historical news explicitly states a decision or agreement is imminent.
- When the historical narrative centers on a crisis, do not assume it will escalate or resolve in the next article; instead, allow for narrative drift toward reporting of earnings, regulatory filings, or even unrelated events not tied to the crisis.
- Avoid generating precise dates or names of pending court rulings, parliamentary debates, or bi-lateral meetings unless the most recent compressed news explicitly confirms the exact timing and topic of such events.

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
- Do not predict decisive legal or regulatory outcomes (e.g., court rulings, arbitration filings, nationalization) unless the most recent historical news explicitly indicates such a decision is imminent; instead, calibrate forecasts to ongoing diplomatic exchanges, industry responses, or policy statements that reflect the current tension level.
- For automaker supply disruptions, avoid assuming specific production halts or shutdowns unless directly confirmed by recent history; true news may cover strategic adaptations (e.g., chip development, financial results, supply resumption) that shift focus away from crisis escalation.
- Ensure event specificity aligns with available signals: do not invent detailed events (e.g., temporary compromise deals, new export bans, parliamentary debates) that lack clear foreshadowing in the preceding compressed news; prefer describing likely next steps for actors already mentioned, such as further talks, operational pivots, or official statements.
- Prefer forecasting financial results, strategic announcements, or industry operational shifts (e.g., chip development, profit warnings) over diplomatic or legal breakthroughs unless the latest historical news explicitly states a decision or agreement is imminent.
- When the historical narrative centers on a crisis, do not assume it will escalate or resolve in the next article; instead, allow for narrative drift toward reporting of earnings, regulatory filings, or even unrelated events not tied to the crisis.
- Avoid generating precise dates or names of pending court rulings, parliamentary debates, or bi-lateral meetings unless the most recent compressed news explicitly confirms the exact timing and topic of such events.
```

## Version 3

- Guidance hash: `125922d7e279`
- Forecast prompt path: `configs\llm_tone\prompts\forecast\baseline.yaml`
- Optimizer prompt path: `configs\llm_tone\prompts\optimizer\anti_overfit_v1.yaml`
- Val RMSE: `3.049818495916064`
- Val MAE: `2.4307588999263463`
- Test RMSE: `2.8430955122512036`
- Test MAE: `2.3331672078174828`

### Guidance

- Do not predict decisive legal or regulatory outcomes (e.g., court rulings, arbitration filings, nationalization) unless the most recent historical news explicitly indicates such a decision is imminent; instead, calibrate forecasts to ongoing diplomatic exchanges, industry responses, or policy statements that reflect the current tension level.
- For automaker supply disruptions, avoid assuming specific production halts or shutdowns unless directly confirmed by recent history; true news may cover strategic adaptations (e.g., chip development, financial results, supply resumption) that shift focus away from crisis escalation.
- Ensure event specificity aligns with available signals: do not invent detailed events (e.g., temporary compromise deals, new export bans, parliamentary debates) that lack clear foreshadowing in the preceding compressed news; prefer describing likely next steps for actors already mentioned, such as further talks, operational pivots, or official statements.
- Prefer forecasting financial results, strategic announcements, or industry operational shifts (e.g., chip development, profit warnings) over diplomatic or legal breakthroughs unless the latest historical news explicitly states a decision or agreement is imminent.
- When the historical narrative centers on a crisis, do not assume it will escalate or resolve in the next article; instead, allow for narrative drift toward reporting of earnings, regulatory filings, or even unrelated events not tied to the crisis.
- Avoid generating precise dates or names of pending court rulings, parliamentary debates, or bi-lateral meetings unless the most recent compressed news explicitly confirms the exact timing and topic of such events.
- Avoid extrapolating political debates, court proceedings, or regulatory decisions unless the most recent history explicitly confirms the date, topic, and participants of such an event; prefer reporting statements, canceled meetings, or ongoing talks.
- When the narrative centers on a supply-chain or geopolitical crisis, allow for sudden topic shifts—true news may report unrelated industry developments (earnings, technology partnerships, executive interviews) instead of continuing the crisis arc; do not force crisis continuity.
- Do not invent specific production adaptations (profit warnings, plant idling percentages, alternative supplier timelines) unless directly stated in recent history; if actors are mentioned, limit forecasts to their confirmed next steps (e.g., seeking partners, issuing warnings, holding talks).

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
- Do not predict decisive legal or regulatory outcomes (e.g., court rulings, arbitration filings, nationalization) unless the most recent historical news explicitly indicates such a decision is imminent; instead, calibrate forecasts to ongoing diplomatic exchanges, industry responses, or policy statements that reflect the current tension level.
- For automaker supply disruptions, avoid assuming specific production halts or shutdowns unless directly confirmed by recent history; true news may cover strategic adaptations (e.g., chip development, financial results, supply resumption) that shift focus away from crisis escalation.
- Ensure event specificity aligns with available signals: do not invent detailed events (e.g., temporary compromise deals, new export bans, parliamentary debates) that lack clear foreshadowing in the preceding compressed news; prefer describing likely next steps for actors already mentioned, such as further talks, operational pivots, or official statements.
- Prefer forecasting financial results, strategic announcements, or industry operational shifts (e.g., chip development, profit warnings) over diplomatic or legal breakthroughs unless the latest historical news explicitly states a decision or agreement is imminent.
- When the historical narrative centers on a crisis, do not assume it will escalate or resolve in the next article; instead, allow for narrative drift toward reporting of earnings, regulatory filings, or even unrelated events not tied to the crisis.
- Avoid generating precise dates or names of pending court rulings, parliamentary debates, or bi-lateral meetings unless the most recent compressed news explicitly confirms the exact timing and topic of such events.
- Avoid extrapolating political debates, court proceedings, or regulatory decisions unless the most recent history explicitly confirms the date, topic, and participants of such an event; prefer reporting statements, canceled meetings, or ongoing talks.
- When the narrative centers on a supply-chain or geopolitical crisis, allow for sudden topic shifts—true news may report unrelated industry developments (earnings, technology partnerships, executive interviews) instead of continuing the crisis arc; do not force crisis continuity.
- Do not invent specific production adaptations (profit warnings, plant idling percentages, alternative supplier timelines) unless directly stated in recent history; if actors are mentioned, limit forecasts to their confirmed next steps (e.g., seeking partners, issuing warnings, holding talks).
```
