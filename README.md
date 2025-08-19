
# Pures Optimizer

  

Build planner and optimizer for Prospecting. 

You can view a live version of this tool at: https://pures-optimizer.streamlit.app/
  

## Features

  

1. Build Planner tab to see total stats and derived timing

2. Optimizer tab to search gear combinations 

3. Monte Carlo Simulations for realistic averages accounting for player times

4. Import and export of build presets as JSON

5. Support for real 6 star values, not just a flat 10% stat difference

  

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


  

## Timing model

  

These constants live near the top of app.py and can be tuned from new measurements.

  

1. K_DIG_FILL controls how many digs are required from capacity and dig strength

2. ALPHA_DIG controls how dig speed scales dig animation time

3. T_DIG_SECONDS base time per dig before dig speed scaling

4. T_SHAKE_SECONDS base time per shake before shake speed scaling

5. FIRST_SHAKE_EXTRA extra one time overhead added at the start of a shake sequence

6. POST_DIG_LOCK_S and POST_SHAKE_LOCK_S fixed lockouts per pan

7. ANIM_EXTRA optional multiplier applied to total time when the animation toggle is on

  

Cycle time equals:

fixed lockouts plus human overhead plus dig time plus shake time

  

Efficiency equals:

effective luck times square root of capacity divided by cycle time after the animation multiplier

  

Profit rate equals:

efficiency times one plus sell boost divided by one hundred

  

## Troubleshooting

  

1. If you change CSV headers or code and Streamlit shows stale behavior, clear the cache or rerun the app.

2. If you see out of bounds errors when indexing rows, check for trailing spaces in CSV name columns.

3. If a widget cannot be modified after creation, ensure imported JSON is applied to session state before any widgets are created. The app already stages pending imports and reruns to apply them.

4. If speed or passives look doubled, confirm that pan passive values are not already baked into the numeric columns. The loader only fills from passives when the numeric column is zero or missing.

  

## Folder layout

Main Folder/

/app.py

/data CSVs

  

## License

  [MIT](https://raw.githubusercontent.com/PureIsntHere/prospecting_build_planner/refs/heads/main/LICENSE)

