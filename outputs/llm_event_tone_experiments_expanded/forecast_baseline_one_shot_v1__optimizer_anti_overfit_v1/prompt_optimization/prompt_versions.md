# Prompt Versions

## Version 0

- Guidance hash: `5677be43abd1`
- Forecast prompt path: `configs\llm_tone\prompts\forecast\baseline_one_shot_v1.yaml`
- Optimizer prompt path: `configs\llm_tone\prompts\optimizer\anti_overfit_v1.yaml`
- Val RMSE: `3.194960623257503`
- Val MAE: `2.538524383643497`
- Test RMSE: `2.8002278275822663`
- Test MAE: `2.421655718994748`

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

- Guidance hash: `b6290dfa7b8a`
- Forecast prompt path: `configs\llm_tone\prompts\forecast\baseline_one_shot_v1.yaml`
- Optimizer prompt path: `configs\llm_tone\prompts\optimizer\anti_overfit_v1.yaml`
- Val RMSE: `3.2062586851009227`
- Val MAE: `2.4912001009508367`
- Test RMSE: `2.515789673817195`
- Test MAE: `2.158586195356199`

### Guidance

- When the historical trend is a dispute or disruption, do not automatically assume the next news will be the most adversarial or disruptive event (e.g., immediate production halts, firm court rulings). Consider that the narrative may pivot to a different actor's response, a tactical de-escalation, or a lateral development (e.g., supplier switch, geopolitical commentary, unrelated news on the same topic).
- For ongoing legal, regulatory, or diplomatic processes, avoid predicting the specific outcome (e.g., 'the court ruled that…') unless strongly supported by scheduled deadlines. Instead, frame the forecast around the likely next actionable development (e.g., 'China intensified its diplomatic pressure' or 'automakers began diversifying supply') without committing to a precise ruling or action date.
- Be alert to topic drift: the next newsworthy event may not advance the same sub-story (e.g., it could be about a different company's strategy, an opinion piece, or a local policy issue). Do not assume the narrative will stay on the most dramatic trajectory. Use the set of recent articles to gauge which dimensions are likely to generate the next headline.

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
- When the historical trend is a dispute or disruption, do not automatically assume the next news will be the most adversarial or disruptive event (e.g., immediate production halts, firm court rulings). Consider that the narrative may pivot to a different actor's response, a tactical de-escalation, or a lateral development (e.g., supplier switch, geopolitical commentary, unrelated news on the same topic).
- For ongoing legal, regulatory, or diplomatic processes, avoid predicting the specific outcome (e.g., 'the court ruled that…') unless strongly supported by scheduled deadlines. Instead, frame the forecast around the likely next actionable development (e.g., 'China intensified its diplomatic pressure' or 'automakers began diversifying supply') without committing to a precise ruling or action date.
- Be alert to topic drift: the next newsworthy event may not advance the same sub-story (e.g., it could be about a different company's strategy, an opinion piece, or a local policy issue). Do not assume the narrative will stay on the most dramatic trajectory. Use the set of recent articles to gauge which dimensions are likely to generate the next headline.
```

## Version 2

- Guidance hash: `b8c53eff23e0`
- Forecast prompt path: `configs\llm_tone\prompts\forecast\baseline_one_shot_v1.yaml`
- Optimizer prompt path: `configs\llm_tone\prompts\optimizer\anti_overfit_v1.yaml`
- Val RMSE: `2.9462697895520806`
- Val MAE: `2.3377627839123036`
- Test RMSE: `3.171213993423352`
- Test MAE: `2.73986976460908`

### Guidance

- When the historical trend is a dispute or disruption, do not automatically assume the next news will be the most adversarial or disruptive event (e.g., immediate production halts, firm court rulings). Consider that the narrative may pivot to a different actor's response, a tactical de-escalation, or a lateral development (e.g., supplier switch, geopolitical commentary, unrelated news on the same topic).
- For ongoing legal, regulatory, or diplomatic processes, avoid predicting the specific outcome (e.g., 'the court ruled that…') unless strongly supported by scheduled deadlines. Instead, frame the forecast around the likely next actionable development (e.g., 'China intensified its diplomatic pressure' or 'automakers began diversifying supply') without committing to a precise ruling or action date.
- Be alert to topic drift: the next newsworthy event may not advance the same sub-story (e.g., it could be about a different company's strategy, an opinion piece, or a local policy issue). Do not assume the narrative will stay on the most dramatic trajectory. Use the set of recent articles to gauge which dimensions are likely to generate the next headline.
- When the latest articles describe ongoing diplomatic pressure, legal proceedings, or corporate posturing, forecast the next newsworthy development as a statement, request, or opinion from a previously secondary actor (e.g., a government spokesman, industry group, or opinion writer) rather than a decisive ruling, investment, or operational change.
- If the sequence includes a scheduled event (e.g., a hearing, visit, or deadline), anchor the forecast on the immediate procedural next step (e.g., 'arguments were heard, a decision is expected within X weeks') and avoid predicting the specific outcome or its downstream effects unless strongly supported by a firm deadline in the latest article.
- Assume topic drift is likely when the most recent article ends on a note of stalemate, internal strife, or diplomatic request: the next headline may come from an entirely different dimension (e.g., a local policy issue, a supplier diversification move, or a broader geopolitical commentary) rather than advancing the same sub-story.

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
- When the historical trend is a dispute or disruption, do not automatically assume the next news will be the most adversarial or disruptive event (e.g., immediate production halts, firm court rulings). Consider that the narrative may pivot to a different actor's response, a tactical de-escalation, or a lateral development (e.g., supplier switch, geopolitical commentary, unrelated news on the same topic).
- For ongoing legal, regulatory, or diplomatic processes, avoid predicting the specific outcome (e.g., 'the court ruled that…') unless strongly supported by scheduled deadlines. Instead, frame the forecast around the likely next actionable development (e.g., 'China intensified its diplomatic pressure' or 'automakers began diversifying supply') without committing to a precise ruling or action date.
- Be alert to topic drift: the next newsworthy event may not advance the same sub-story (e.g., it could be about a different company's strategy, an opinion piece, or a local policy issue). Do not assume the narrative will stay on the most dramatic trajectory. Use the set of recent articles to gauge which dimensions are likely to generate the next headline.
- When the latest articles describe ongoing diplomatic pressure, legal proceedings, or corporate posturing, forecast the next newsworthy development as a statement, request, or opinion from a previously secondary actor (e.g., a government spokesman, industry group, or opinion writer) rather than a decisive ruling, investment, or operational change.
- If the sequence includes a scheduled event (e.g., a hearing, visit, or deadline), anchor the forecast on the immediate procedural next step (e.g., 'arguments were heard, a decision is expected within X weeks') and avoid predicting the specific outcome or its downstream effects unless strongly supported by a firm deadline in the latest article.
- Assume topic drift is likely when the most recent article ends on a note of stalemate, internal strife, or diplomatic request: the next headline may come from an entirely different dimension (e.g., a local policy issue, a supplier diversification move, or a broader geopolitical commentary) rather than advancing the same sub-story.
```

## Version 3

- Guidance hash: `b22c1efea2be`
- Forecast prompt path: `configs\llm_tone\prompts\forecast\baseline_one_shot_v1.yaml`
- Optimizer prompt path: `configs\llm_tone\prompts\optimizer\anti_overfit_v1.yaml`
- Val RMSE: `3.2419216880472357`
- Val MAE: `2.6096637089559724`
- Test RMSE: `2.7251432593485836`
- Test MAE: `2.269853890289328`

### Guidance

- When the historical trend is a dispute or disruption, do not automatically assume the next news will be the most adversarial or disruptive event (e.g., immediate production halts, firm court rulings). Consider that the narrative may pivot to a different actor's response, a tactical de-escalation, or a lateral development (e.g., supplier switch, geopolitical commentary, unrelated news on the same topic).
- For ongoing legal, regulatory, or diplomatic processes, avoid predicting the specific outcome (e.g., 'the court ruled that…') unless strongly supported by scheduled deadlines. Instead, frame the forecast around the likely next actionable development (e.g., 'China intensified its diplomatic pressure' or 'automakers began diversifying supply') without committing to a precise ruling or action date.
- Be alert to topic drift: the next newsworthy event may not advance the same sub-story (e.g., it could be about a different company's strategy, an opinion piece, or a local policy issue). Do not assume the narrative will stay on the most dramatic trajectory. Use the set of recent articles to gauge which dimensions are likely to generate the next headline.
- When the latest articles describe ongoing diplomatic pressure, legal proceedings, or corporate posturing, forecast the next newsworthy development as a statement, request, or opinion from a previously secondary actor (e.g., a government spokesman, industry group, or opinion writer) rather than a decisive ruling, investment, or operational change.
- If the sequence includes a scheduled event (e.g., a hearing, visit, or deadline), anchor the forecast on the immediate procedural next step (e.g., 'arguments were heard, a decision is expected within X weeks') and avoid predicting the specific outcome or its downstream effects unless strongly supported by a firm deadline in the latest article.
- Assume topic drift is likely when the most recent article ends on a note of stalemate, internal strife, or diplomatic request: the next headline may come from an entirely different dimension (e.g., a local policy issue, a supplier diversification move, or a broader geopolitical commentary) rather than advancing the same sub-story.
- When the latest articles depict a diplomatic stalemate, internal corporate spat, or defensive posturing, do not assume the next headline will advance the same sub-story; treat topic drift as the default expectation—the next news may be about a different company's financials, an unrelated local event, or a competitive shift (e.g., EV market threats) that does not directly address the dispute.
- Eschew predicting concrete government actions (e.g., invoking regulations, issuing specific ministerial statements) or corporate investments unless the latest article provides a firm deadline or explicit imminent plan; instead, anchor forecasts on statements from secondary actors (industry associations, lawmakers, analysts) or factual updates on existing procedural steps (e.g., hearing occurs, meeting called) without assigning a specific outcome.
- If the historical trajectory includes accusations, canceled visits, or public warnings, anticipate that the next newsworthy development may be an escalation from a previously peripheral actor (e.g., a corporate warning with a deadline, a diplomatic demand) rather than a de-escalation or collaborative move; avoid predicting diplomatic progress unless the latest article explicitly signals goodwill gestures.

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
- When the historical trend is a dispute or disruption, do not automatically assume the next news will be the most adversarial or disruptive event (e.g., immediate production halts, firm court rulings). Consider that the narrative may pivot to a different actor's response, a tactical de-escalation, or a lateral development (e.g., supplier switch, geopolitical commentary, unrelated news on the same topic).
- For ongoing legal, regulatory, or diplomatic processes, avoid predicting the specific outcome (e.g., 'the court ruled that…') unless strongly supported by scheduled deadlines. Instead, frame the forecast around the likely next actionable development (e.g., 'China intensified its diplomatic pressure' or 'automakers began diversifying supply') without committing to a precise ruling or action date.
- Be alert to topic drift: the next newsworthy event may not advance the same sub-story (e.g., it could be about a different company's strategy, an opinion piece, or a local policy issue). Do not assume the narrative will stay on the most dramatic trajectory. Use the set of recent articles to gauge which dimensions are likely to generate the next headline.
- When the latest articles describe ongoing diplomatic pressure, legal proceedings, or corporate posturing, forecast the next newsworthy development as a statement, request, or opinion from a previously secondary actor (e.g., a government spokesman, industry group, or opinion writer) rather than a decisive ruling, investment, or operational change.
- If the sequence includes a scheduled event (e.g., a hearing, visit, or deadline), anchor the forecast on the immediate procedural next step (e.g., 'arguments were heard, a decision is expected within X weeks') and avoid predicting the specific outcome or its downstream effects unless strongly supported by a firm deadline in the latest article.
- Assume topic drift is likely when the most recent article ends on a note of stalemate, internal strife, or diplomatic request: the next headline may come from an entirely different dimension (e.g., a local policy issue, a supplier diversification move, or a broader geopolitical commentary) rather than advancing the same sub-story.
- When the latest articles depict a diplomatic stalemate, internal corporate spat, or defensive posturing, do not assume the next headline will advance the same sub-story; treat topic drift as the default expectation—the next news may be about a different company's financials, an unrelated local event, or a competitive shift (e.g., EV market threats) that does not directly address the dispute.
- Eschew predicting concrete government actions (e.g., invoking regulations, issuing specific ministerial statements) or corporate investments unless the latest article provides a firm deadline or explicit imminent plan; instead, anchor forecasts on statements from secondary actors (industry associations, lawmakers, analysts) or factual updates on existing procedural steps (e.g., hearing occurs, meeting called) without assigning a specific outcome.
- If the historical trajectory includes accusations, canceled visits, or public warnings, anticipate that the next newsworthy development may be an escalation from a previously peripheral actor (e.g., a corporate warning with a deadline, a diplomatic demand) rather than a de-escalation or collaborative move; avoid predicting diplomatic progress unless the latest article explicitly signals goodwill gestures.
```
