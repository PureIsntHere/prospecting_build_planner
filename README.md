# Pures Optimizer

Build planner and optimizer for Prospecting

## Features

1. Build Planner tab to see total stats and derived timing
2. Optimizer tab to search gear combinations
3. Monte Carlo option to model human overhead variance
4. Import and export of build presets as JSON
5. Support for six star item overrides and pan passive parsing

## Requirements

1. Python 3.10 or newer
2. Packages: streamlit, pandas, numpy

Install with:
```
pip install streamlit pandas numpy
```

## Running

From the project folder:
```
streamlit run app.py
```

## Data files

Place these CSV files in the project root:

1. equipment.csv
2. equipment_6_full.csv optional
3. equipment_6.csv optional fallback if the full file is not present
4. enchants.csv
5. potions.csv
6. buffs.csv
7. pans.csv
8. shovels.csv

Notes:

1. Names are trimmed on load. Make sure names match across files exactly.
2. Pans may include a passives column with text like Size 25 or Modifier 10 or Sell 15. If size, modifier, or sell are zero or missing, the passive text is parsed and applied once.
3. Multipliers default to 1.0 during load. Other numeric stats default to 0.0.

## Using the app

1. Select a pan and shovel in the sidebar.
2. Pick your necklace, charm, and up to eight rings. Enable six star where applicable.
3. Choose an enchant, potions, and buffs.
4. Adjust toggles such as meteor, totems, and animation time.
5. Review totals and derived values such as digs, shakes, and cycle time. Efficiency and profit rate are shown for quick comparison.

## Optimizer

1. In the Optimizer tab, choose a single pan and shovel to explore.
2. Select allowed enchants and optionally trim the necklace, charm, and ring pools.
3. Set search budgets and jitter. Higher budgets explore more but take longer.
4. Run Optimizer to see top candidates. Optionally verify the top results with Monte Carlo and download the best build JSON.

## Import and export

Export

1. Use Export current build to download a JSON preset.

Import

1. Paste a JSON preset in Import build and submit.
2. The app stores the payload and reruns so widgets initialize from the imported state.


Cycle time equals:
fixed lockouts plus human overhead plus dig time plus shake time

Efficiency equals:
effective luck times square root of capacity divided by cycle time after the animation multiplier

Profit rate equals:
efficiency times one plus sell boost divided by one hundred

## Folder layout

1. app.py main Streamlit application
2. data CSVs in the same folder as app.py

## License
[MIT](https://raw.githubusercontent.com/PureIsntHere/prospecting_build_planner/refs/heads/main/LICENSE)
