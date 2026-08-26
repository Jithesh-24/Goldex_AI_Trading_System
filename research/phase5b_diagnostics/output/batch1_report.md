# GOLEX V3 Phase 5 Batch 1 — Diagnostic Foundation Report

## Horizon h=15

- D1 Direction point-biserial: {'r': 0.03907651351885297, 'n': 252662, 'ci_lo': 0.03518257581262049, 'ci_hi': 0.0429692647595744}
- D2 base rate: {'n': 311077, 'up_frac': 0.48219251182183187, 'down_frac': 0.4842659534456099, 'timeout_frac': 0.03354153473255818}
- D3 Opportunity point-biserial: {'r': 0.19086219043211092, 'n': 210972, 'ci_lo': 0.18674707549031372, 'ci_hi': 0.19497060772441513}, win_rate=0.5149688110270557
- D4 contradiction rates: {'rate': 0.3584267106535464, 'k': 75618, 'n': 210972, 'ci_lo': np.float64(0.356383016470164), 'ci_hi': np.float64(0.3604755605736638)}, {'rate': 0.0, 'k': 0, 'n': 210972, 'ci_lo': np.float64(0.0), 'ci_hi': np.float64(1.8208719874588714e-05)}
- D5 global calibration: {'intercept': 0.07337238978830182, 'slope': 0.013536416717010074, 'intercept_se': 0.004401025634603755, 'slope_se': 0.011363625607768513, 'n': 210972}, traded_subset=computed
- D6: {'horizon': 15, 'direction': {'long': {'r': 0.018954489622413872, 'n': 126639, 'ci_lo': 0.01344815790916993, 'ci_hi': 0.024459671774151974}, 'short': {'r': 0.008426385951390286, 'n': 126023, 'ci_lo': 0.0029053409325688127, 'ci_hi': 0.013946917276384101}}, 'opportunity': {'long': {'r': 0.007342564159193161, 'n': 104414, 'ci_lo': 0.0012769672556402509, 'ci_hi': 0.01340762079434611}, 'short': {'r': -0.003036386814789118, 'n': 106558, 'ci_lo': -0.009040543452991547, 'ci_hi': 0.0029679887562779604}}, 'barrier': {'long': {'r': 0.007342564159193161, 'n': 104414, 'ci_lo': 0.0012769672556402509, 'ci_hi': 0.01340762079434611}, 'short': {'r': -0.003036386814789118, 'n': 106558, 'ci_lo': -0.009040543452991547, 'ci_hi': 0.0029679887562779604}}}

### Attribution
- **market/labels** (not decisive): D2 raw label base rate: up=0.48219251182183187, down=0.4842659534456099 (dominant-side frac=0.4843)
- **direction** (not decisive): D1 Direction point-biserial r=0.03907651351885297 (n=252662)
- **downstream_specialists** (not decisive): D3 Opportunity point-biserial r=0.19086219043211092 vs D1 Direction r=0.03907651351885297
- **calibration** (DECISIVE): D5 global calibration slope=0.013536416717010074 (ideal=1.0), traded_subset=computed
- **disagreement** (DECISIVE): D4 contradiction rates: barrier_vs_reward_risk=0.3584267106535464, opportunity_vs_barrier=0.0

## Horizon h=45

- D1 Direction point-biserial: {'r': 0.028858981212295843, 'n': 250495, 'ci_lo': 0.024945672074774963, 'ci_hi': 0.03277140592078519}
- D2 base rate: {'n': 311074, 'up_frac': 0.47978937487543155, 'down_frac': 0.4790596449719359, 'timeout_frac': 0.041150980152632494}
- D3 Opportunity point-biserial: {'r': 0.19929336698690675, 'n': 209128, 'ci_lo': 0.19517409553619897, 'ci_hi': 0.20340560734392374}, win_rate=0.49945009754791325
- D4 contradiction rates: {'rate': 0.0834417199035997, 'k': 17450, 'n': 209128, 'ci_lo': np.float64(0.08226407608294892), 'ci_hi': np.float64(0.08463466747021836)}, {'rate': 0.0, 'k': 0, 'n': 209128, 'ci_lo': np.float64(0.0), 'ci_hi': np.float64(1.836927351956145e-05)}
- D5 global calibration: {'intercept': 0.05189212095162599, 'slope': -0.023392819547686763, 'intercept_se': 0.004375909720742281, 'slope_se': 0.010463836790703429, 'n': 209128}, traded_subset=N/A
- D6: {'horizon': 45, 'direction': {'long': {'r': 0.014842459801284454, 'n': 107486, 'ci_lo': 0.008864900950691645, 'ci_hi': 0.020818957928329097}, 'short': {'r': 0.010697363768087699, 'n': 143009, 'ci_lo': 0.005514742991647527, 'ci_hi': 0.01587940988955866}}, 'opportunity': {'long': {'r': 0.04522877641106286, 'n': 84825, 'ci_lo': 0.03851080504209655, 'ci_hi': 0.05194265944879088}, 'short': {'r': 0.020752331686757816, 'n': 124303, 'ci_lo': 0.015194836873489766, 'ci_hi': 0.026308544340998833}}, 'barrier': {'long': {'r': 0.04522877641106286, 'n': 84825, 'ci_lo': 0.03851080504209655, 'ci_hi': 0.05194265944879088}, 'short': {'r': 0.020752331686757816, 'n': 124303, 'ci_lo': 0.015194836873489766, 'ci_hi': 0.026308544340998833}}}

### Attribution
- **market/labels** (not decisive): D2 raw label base rate: up=0.47978937487543155, down=0.4790596449719359 (dominant-side frac=0.4798)
- **direction** (not decisive): D1 Direction point-biserial r=0.028858981212295843 (n=250495)
- **downstream_specialists** (not decisive): D3 Opportunity point-biserial r=0.19929336698690675 vs D1 Direction r=0.028858981212295843
- **calibration** (DECISIVE): D5 global calibration slope=-0.023392819547686763 (ideal=1.0), traded_subset=N/A (zero trades at this horizon)
- **disagreement** (not decisive): D4 contradiction rates: barrier_vs_reward_risk=0.0834417199035997, opportunity_vs_barrier=0.0

## Horizon h=90

- D1 Direction point-biserial: {'r': 0.023589040212981974, 'n': 248087, 'ci_lo': 0.019655776732755866, 'ci_hi': 0.027521573551014047}
- D2 base rate: {'n': 311064, 'up_frac': 0.47595671630275443, 'down_frac': 0.4740117789265232, 'timeout_frac': 0.05003150477072242}
- D3 Opportunity point-biserial: {'r': 0.22038277853810495, 'n': 207109, 'ci_lo': 0.21628123858548173, 'ci_hi': 0.2244765399231882}, win_rate=0.405477309049824
- D4 contradiction rates: {'rate': 0.01173778058896523, 'k': 2431, 'n': 207109, 'ci_lo': np.float64(0.011282894276802239), 'ci_hi': np.float64(0.012210779811246195)}, {'rate': 0.0, 'k': 0, 'n': 207109, 'ci_lo': np.float64(0.0), 'ci_hi': np.float64(1.854834287590596e-05)}
- D5 global calibration: {'intercept': 0.02924353086375969, 'slope': -0.015432614663491112, 'intercept_se': 0.005690210820694996, 'slope_se': 0.00840572708169672, 'n': 207109}, traded_subset=N/A
- D6: {'horizon': 90, 'direction': {'long': {'r': 0.01022860518551032, 'n': 112060, 'ci_lo': 0.004373803583837236, 'ci_hi': 0.016082705551192244}, 'short': {'r': 0.017655225114067814, 'n': 136027, 'ci_lo': 0.012342105811845382, 'ci_hi': 0.02296734750551711}}, 'opportunity': {'long': {'r': 0.11837415678038207, 'n': 89813, 'ci_lo': 0.11192065669227727, 'ci_hi': 0.12481767219996609}, 'short': {'r': 0.07722561941573806, 'n': 117296, 'ci_lo': 0.07153434580046239, 'ci_hi': 0.08291186469549344}}, 'barrier': {'long': {'r': 0.11837415678038207, 'n': 89813, 'ci_lo': 0.11192065669227727, 'ci_hi': 0.12481767219996609}, 'short': {'r': 0.07722561941573806, 'n': 117296, 'ci_lo': 0.07153434580046239, 'ci_hi': 0.08291186469549344}}}

### Attribution
- **market/labels** (not decisive): D2 raw label base rate: up=0.47595671630275443, down=0.4740117789265232 (dominant-side frac=0.4760)
- **direction** (not decisive): D1 Direction point-biserial r=0.023589040212981974 (n=248087)
- **downstream_specialists** (not decisive): D3 Opportunity point-biserial r=0.22038277853810495 vs D1 Direction r=0.023589040212981974
- **calibration** (DECISIVE): D5 global calibration slope=-0.015432614663491112 (ideal=1.0), traded_subset=N/A (zero trades at this horizon)
- **disagreement** (not decisive): D4 contradiction rates: barrier_vs_reward_risk=0.01173778058896523, opportunity_vs_barrier=0.0
