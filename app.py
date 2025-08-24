import math, json, time
import numpy as np
import pandas as pd
import streamlit as st

APP_TITLE = "Pures Optimizer"

ANIM_EXTRA = 0.20  # +20% time when animation calibration is on

# Timing model
K_DIG_FILL = 1.5

# Digging
ALPHA_DIG = 0.50
T_DIG_SECONDS = 1.40

# Shaking
T_SHAKE_SECONDS = 0.54
FIRST_SHAKE_EXTRA = 0.07

# Lockouts (fixed per pan)
POST_DIG_LOCK_S = 0.16
POST_SHAKE_LOCK_S = 1.95
BASE_OVERHEAD_S = POST_DIG_LOCK_S + POST_SHAKE_LOCK_S


# Load Data CSVs
@st.cache_data
def load_csvs():
    equip = pd.read_csv("equipment.csv")

    # Optional 6★ overrides
    equip6_full = None
    try:
        equip6_full = pd.read_csv("equipment_6_full.csv")
    except Exception:
        try:
            equip6_full = pd.read_csv("equipment_6.csv")
        except Exception:
            equip6_full = None

    # --- Normalize column names early ---
    equip.columns = [c.strip() for c in equip.columns]
    if equip6_full is not None:
        equip6_full.columns = [c.strip() for c in equip6_full.columns]

    # --- Trim names so index lookups are stable ---
    if "name" in equip.columns:
        equip["name"] = equip["name"].astype(str).str.strip()
    if equip6_full is not None and "name" in equip6_full.columns:
        equip6_full["name"] = equip6_full["name"].astype(str).str.strip()

    # Build 6★ override map (store replacement values for differing numeric cols)
    equip6_map = {}
    if equip6_full is not None:
        numeric_cols = [
            "luck",
            "dig_str",
            "capacity",
            "dig_speed",
            "shake_str",
            "shake_speed",
            "sell",
            "size",
            "modifier",
        ]
        base_by_name = equip.set_index("name")
        for _, r in equip6_full.iterrows():
            name = str(r.get("name", "")).strip()
            if not name or name not in base_by_name.index:
                continue
            b = base_by_name.loc[name]
            diffs = {}
            for col in numeric_cols:
                try:
                    v6 = float(r.get(col, np.nan))
                    vb = float(b.get(col, np.nan))
                except Exception:
                    continue
                if pd.isna(v6) or pd.isna(vb):
                    continue
                if abs(v6 - vb) > 1e-9:
                    diffs[col] = v6  # store absolute replacement, not delta
            if diffs:
                equip6_map[name] = diffs

    # Other CSVs  (the typo was here — must be read_csv)
    ench = pd.read_csv("enchants.csv")
    pots = pd.read_csv("potions.csv")
    buffs = pd.read_csv("buffs.csv")
    pans = pd.read_csv("pans.csv")
    shovels = pd.read_csv("shovels.csv")

    # Trim 'name' everywhere to avoid whitespace mismatches
    for _df in (ench, pots, buffs, pans, shovels):
        if _df is not None and "name" in _df.columns:
            _df["name"] = _df["name"].astype(str).str.strip()

    # Ensure required columns exist (use NaN so we can choose defaults later)
    for df, cols in [
        (
            pans,
            [
                "name",
                "luck",
                "capacity",
                "shake_str",
                "shake_speed_mult",
                "sell",
                "size",
                "modifier",
                "passives",
            ],
        ),
        (
            shovels,
            [
                "name",
                "dig_str",
                "dig_speed_mult",
                "toughness",
                "sell",
                "size",
                "modifier",
            ],
        ),
    ]:
        for c in cols:
            if c not in df.columns:
                df[c] = "" if c in ("name", "passives") else np.nan

    # Numeric coercion with correct defaults:
    # - multipliers -> default 1.0
    # - others      -> default 0.0
    def _num(s, default):
        return pd.to_numeric(s, errors="coerce").fillna(default)

    pans["luck"] = _num(pans["luck"], 0.0)
    pans["capacity"] = _num(pans["capacity"], 0.0)
    pans["shake_str"] = _num(pans["shake_str"], 0.0)
    pans["shake_speed_mult"] = _num(pans["shake_speed_mult"], 1.0)
    pans["sell"] = _num(pans["sell"], 0.0)
    pans["size"] = _num(pans["size"], 0.0)
    pans["modifier"] = _num(pans["modifier"], 0.0)

    shovels["dig_str"] = _num(shovels["dig_str"], 0.0)
    shovels["dig_speed_mult"] = _num(shovels["dig_speed_mult"], 1.0)
    shovels["toughness"] = _num(shovels["toughness"], 0.0)
    shovels["sell"] = _num(shovels["sell"], 0.0)
    shovels["size"] = _num(shovels["size"], 0.0)
    shovels["modifier"] = _num(shovels["modifier"], 0.0)

    # --- Parse pan passive text into sell/size/modifier IF those fields are blank/zero ---
    import re

    def _parse_passives(text: str):
        sell = size = modifier = 0.0
        if isinstance(text, str) and text.strip():
            t = text.lower().replace("boost", "").replace("%", "")
            for kind, val in re.findall(
                r"(size|modifier|sell)\s*([+\-]?\d+(?:\.\d+)?)", t
            ):
                v = float(val)
                if kind == "size":
                    size += v
                elif kind == "modifier":
                    modifier += v
                elif kind == "sell":
                    sell += v
        return sell, size, modifier

    if "passives" in pans.columns:
        for i, txt in pans["passives"].items():
            se, si, mo = _parse_passives(txt)
            if not np.isfinite(pans.loc[i, "sell"]) or pans.loc[i, "sell"] == 0.0:
                pans.loc[i, "sell"] = se
            if not np.isfinite(pans.loc[i, "size"]) or pans.loc[i, "size"] == 0.0:
                pans.loc[i, "size"] = si
            if (
                not np.isfinite(pans.loc[i, "modifier"])
                or pans.loc[i, "modifier"] == 0.0
            ):
                pans.loc[i, "modifier"] = mo

    return equip, equip6_map, ench, pots, buffs, pans, shovels


def row_to_item(row, star6, overrides=None):
    return dict(
        name=row["name"],
        luck=float(row.get("luck", 0.0)),
        dig_str=float(row.get("dig_str", 0.0)),
        capacity=float(row.get("capacity", 0.0)),
        dig_speed=float(row.get("dig_speed", 0.0)),
        shake_str=float(row.get("shake_str", 0.0)),
        shake_speed=float(row.get("shake_speed", 0.0)),
        sell=float(row.get("sell", 0.0)),
        size=float(row.get("size", 0.0)),
        modifier=float(row.get("modifier", 0.0)),
        star6=bool(star6),
        overrides=(overrides or {}),
    )


# Computation
def compute_once(
    pan,
    shovel,
    items,
    ench_row,
    potion_rows,
    buff_rows,
    luck_mult,
    str_mult,
    overhead_s,
    login_bonus_luck,
    anim_mult: float = 1.0,
):
    totals = dict(
        luck=float(pan["luck"]),
        capacity=float(pan["capacity"]),
        shake_str=float(pan["shake_str"]),
        dig_str=float(shovel["dig_str"]),
        dig_speed=0.0,
        shake_speed=0.0,
        sell=float(pan.get("sell", 0.0)) + float(shovel.get("sell", 0.0)),
        size=float(pan.get("size", 0.0)) + float(shovel.get("size", 0.0)),
        modifier=float(pan.get("modifier", 0.0)) + float(shovel.get("modifier", 0.0)),
    )

    # Equipment (apply 6★ overrides if present on that item)
    for it in items:
        use_over = (
            bool(it.get("star6"))
            and isinstance(it.get("overrides"), dict)
            and len(it["overrides"]) > 0
        )

        def v(key):
            if use_over and key in it["overrides"]:
                return float(it["overrides"][key])
            return float(it.get(key, 0.0))

        totals["luck"] += v("luck")
        totals["dig_str"] += v("dig_str")
        totals["capacity"] += v("capacity")
        totals["dig_speed"] += v("dig_speed")
        totals["shake_str"] += v("shake_str")
        totals["shake_speed"] += v("shake_speed")
        totals["sell"] += v("sell")
        totals["size"] += v("size")
        totals["modifier"] += v("modifier")

    # Enchant
    if ench_row is not None and len(ench_row):
        if isinstance(ench_row, pd.Series):
            ench_row = ench_row.to_dict()
        totals["luck"] += float(ench_row.get("luck", 0.0))
        totals["capacity"] += float(ench_row.get("capacity", 0.0))
        totals["shake_str"] += float(ench_row.get("shake_str", 0.0))
        totals["shake_speed"] += float(ench_row.get("shake_speed", 0.0))
        totals["sell"] += float(ench_row.get("sell", 0.0))
        totals["size"] += float(ench_row.get("size", 0.0))
        totals["modifier"] += float(ench_row.get("modifier", 0.0))

    # Potions
    for pr in potion_rows:
        if isinstance(pr, pd.Series):
            pr = pr.to_dict()
        totals["luck"] += float(pr.get("luck", 0.0))
        totals["dig_str"] += float(pr.get("dig_str", 0.0))
        totals["capacity"] += float(pr.get("capacity", 0.0))
        totals["dig_speed"] += float(pr.get("dig_speed", 0.0))
        totals["shake_str"] += float(pr.get("shake_str", 0.0))
        totals["shake_speed"] += float(pr.get("shake_speed", 0.0))
        totals["sell"] += float(pr.get("sell", 0.0))
        totals["size"] += float(pr.get("size", 0.0))
        totals["modifier"] += float(pr.get("modifier", 0.0))

    # Buffs
    for br in buff_rows:
        if isinstance(br, pd.Series):
            br = br.to_dict()
        totals["luck"] += float(br.get("luck", 0.0))
        totals["dig_str"] += float(br.get("dig_str", 0.0))
        totals["capacity"] += float(br.get("capacity", 0.0))
        totals["dig_speed"] += float(br.get("dig_speed", 0.0))
        totals["shake_str"] += float(br.get("shake_str", 0.0))
        totals["shake_speed"] += float(br.get("shake_speed", 0.0))
        totals["sell"] += float(br.get("sell", 0.0))
        totals["size"] += float(br.get("size", 0.0))
        totals["modifier"] += float(br.get("modifier", 0.0))

    # Login bonus
    totals["luck"] += float(login_bonus_luck)

    # Effective multipliers
    eff_luck = totals["luck"] * float(luck_mult)
    eff_shake_str = totals["shake_str"] * float(str_mult)

    # Speed multipliers
    dig_factor = (1.0 + totals["dig_speed"] / 100.0) * float(shovel["dig_speed_mult"])
    shake_factor = 1.0 + totals["shake_speed"] / 100.0

    # Digs: number of shovel actions needed to fill the pan
    # digs ≈ ceil( Capacity / ( K * DigStrength ) )
    digs = int(
        np.ceil(
            max(1e-6, totals["capacity"]) / max(1e-6, K_DIG_FILL * totals["dig_str"])
        )
    )
    # Calibrated dig animation law: t_per_dig = T_DIG_SECONDS / (dig_factor ** ALPHA_DIG)
    dig_time = (digs * float(T_DIG_SECONDS)) / (
        max(1e-6, dig_factor) ** float(ALPHA_DIG)
    )

    # Shakes: number needed to process the dirt
    # shakes ≈ ceil( Capacity / ShakeStrength )
    shakes = int(np.ceil(max(1e-6, totals["capacity"]) / max(1e-6, eff_shake_str)))
    per_shake_seconds = float(T_SHAKE_SECONDS) * float(pan["shake_speed_mult"])
    shake_time_core = (shakes * per_shake_seconds) / max(1e-6, shake_factor)

    # Add a small startup penalty for the *first* shake (constant; not scaled by speed)
    first_shake_extra = float(globals().get("FIRST_SHAKE_EXTRA", 0.0))
    shake_time = shake_time_core + (first_shake_extra if shakes > 0 else 0.0)

    # Total time per pan
    # Fixed lockouts (post-dig + post-shake) + any user/MC overhead passed in
    total_time = float(BASE_OVERHEAD_S) + float(overhead_s) + dig_time + shake_time

    # Metrics with animation calibration applied to time
    time_with_anim = max(1e-6, total_time) * float(anim_mult)
    efficiency = (eff_luck * math.sqrt(max(0.0, totals["capacity"]))) / time_with_anim
    profit_rate = efficiency * (1.0 + totals["sell"] / 100.0)

    return totals, dict(
        eff_luck=eff_luck,
        eff_shake_str=eff_shake_str,
        shakes=shakes,
        digs=digs,
        time_per_pan_s=total_time,  # raw cycle time (without anim multiplier)
        efficiency=efficiency,
        profit_rate=profit_rate,
    )


def mc_overhead_sim(
    pan,
    shovel,
    items,
    ench_row,
    potion_rows,
    buff_rows,
    luck_mult,
    str_mult,
    overhead_mu,
    overhead_sigma,
    overhead_min,
    overhead_max,
    login_bonus_luck,
    runs=400,
    seed=42,
    anim_mult: float = 1.0,
):
    """
    Monte Carlo on per-pan cycle time using the same timing law as compute_once:
      - digs: t = digs * T_DIG_SECONDS / (dig_factor ** ALPHA_DIG)
      - shakes: t = (shakes * T_SHAKE_SECONDS * pan_mult) / shake_factor + FIRST_SHAKE_EXTRA
      - fixed lockouts: BASE_OVERHEAD_S
      - extra human delay: ~N(mu, sigma) clamped to [min, max]
      - efficiency/profit computed with animation multiplier applied to time
    """
    # Build baseline with 0 extra overhead so we don't double-count it in trials
    base_totals, base_res = compute_once(
        pan,
        shovel,
        items,
        ench_row,
        potion_rows,
        buff_rows,
        luck_mult,
        str_mult,
        0.0,  # no user overhead here
        login_bonus_luck,
        anim_mult=1.0,  # baseline; anim is applied per MC sample
    )

    # Speed factors (constant across all samples for a fixed build)
    dig_factor = (1.0 + float(base_totals.get("dig_speed", 0.0)) / 100.0) * float(
        shovel.get("dig_speed_mult", 1.0)
    )
    shake_factor = 1.0 + float(base_totals.get("shake_speed", 0.0)) / 100.0
    dig_factor = max(1e-6, dig_factor)
    shake_factor = max(1e-6, shake_factor)

    # Counts from the same law as compute_once
    digs = int(base_res["digs"])
    shakes = int(base_res["shakes"])

    # Per-shake seconds + first-shake constant add-on
    pan_mult = float(pan.get("shake_speed_mult", 1.0))
    per_shake_seconds = float(T_SHAKE_SECONDS) * pan_mult
    first_shake_extra = float(globals().get("FIRST_SHAKE_EXTRA", 0.0))

    # Scalars for efficiency/profit
    eff_luck = float(base_res["eff_luck"])
    cap = float(base_totals["capacity"])
    sell = float(base_totals["sell"])

    rng = np.random.default_rng(seed)
    eff_vals, prof_vals, time_vals = [], [], []

    for _ in range(int(runs)):
        # Sample extra human delay and clamp
        extra = float(
            np.clip(rng.normal(overhead_mu, overhead_sigma), overhead_min, overhead_max)
        )

        # Multi-dig timing with exponent
        dig_time = (digs * float(T_DIG_SECONDS)) / (dig_factor ** float(ALPHA_DIG))

        # Shake timing + constant first-shake startup
        shake_time_core = (shakes * per_shake_seconds) / shake_factor
        shake_time = shake_time_core + (first_shake_extra if shakes > 0 else 0.0)

        # Fixed lockouts + sampled human delay
        total_time = float(BASE_OVERHEAD_S) + extra + dig_time + shake_time

        # Metrics (apply animation multiplier to time)
        time_with_anim = max(1e-6, total_time) * float(anim_mult)
        eff = (eff_luck * math.sqrt(max(0.0, cap))) / time_with_anim
        prof = eff * (1.0 + sell / 100.0)

        time_vals.append(total_time)
        eff_vals.append(eff)
        prof_vals.append(prof)

    return dict(
        time_mean=float(np.mean(time_vals)),
        time_std=float(np.std(time_vals, ddof=1)),
        eff_mean=float(np.mean(eff_vals)),
        eff_std=float(np.std(eff_vals, ddof=1)),
        prof_mean=float(np.mean(prof_vals)),
        prof_std=float(np.std(prof_vals, ddof=1)),
    )


def empty_item(slot_name="(Empty)"):
    return dict(
        name=slot_name,
        luck=0.0,
        dig_str=0.0,
        capacity=0.0,
        dig_speed=0.0,
        shake_str=0.0,
        shake_speed=0.0,
        sell=0.0,
        size=0.0,
        modifier=0.0,
        star6=False,
        overrides={},
    )


def score_from_res(res, objective):
    return res["efficiency"] if objective.startswith("Mythic") else res["profit_rate"]


# UI
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
st.caption("Build Planner and Build Generator")

equip, equip6_map, enchants, potions, buffs, pans, shovels = load_csvs()

# === Apply a pending imported build BEFORE any widgets are created ===
pending = st.session_state.pop("pending_import_payload", None)
if pending:
    # Make quick name lists for membership checks
    _neck_names = (
        equip[equip["slot"].astype(str).str.strip().str.lower() == "necklace"]["name"]
        .astype(str)
        .str.strip()
        .tolist()
    )
    _charm_names = (
        equip[equip["slot"].astype(str).str.strip().str.lower() == "charm"]["name"]
        .astype(str)
        .str.strip()
        .tolist()
    )
    _ring_names = (
        equip[equip["slot"].astype(str).str.strip().str.lower() == "ring"]["name"]
        .astype(str)
        .str.strip()
        .tolist()
    )
    _pan_names = pans["name"].astype(str).str.strip().tolist()
    _shv_names = shovels["name"].astype(str).str.strip().tolist()

    # Pan / Shovel
    pn = str(pending.get("pan_name", "")).strip()
    sv = str(pending.get("shovel_name", "")).strip()
    if pn in _pan_names:
        st.session_state["pan_select"] = pn
    if sv in _shv_names:
        st.session_state["shovel_select"] = sv

    # Neck
    neck = pending.get("neck") or {}
    nn = str(neck.get("name", "(Empty)")).strip()
    st.session_state["neck_select"] = (
        nn if (nn == "(Empty)" or nn in _neck_names) else "(Empty)"
    )
    st.session_state["neck_star6_cb"] = bool(
        neck.get("star6", False) and nn in equip6_map
    )

    # Charm
    charm = pending.get("charm") or {}
    cn = str(charm.get("name", "(Empty)")).strip()
    st.session_state["charm_select"] = (
        cn if (cn == "(Empty)" or cn in _charm_names) else "(Empty)"
    )
    st.session_state["charm_star6_cb"] = bool(
        charm.get("star6", False) and cn in equip6_map
    )

    # Rings (8)
    rings = pending.get("rings") or []
    for i in range(8):
        r = (
            rings[i]
            if (i < len(rings) and isinstance(rings[i], dict))
            else {"name": "(Empty)", "star6": False}
        )
        rn = str(r.get("name", "(Empty)")).strip()
        st.session_state[f"ring_sel_{i+1}"] = (
            rn if (rn == "(Empty)" or rn in _ring_names) else "(Empty)"
        )
        st.session_state[f"ring_star6_{i+1}"] = bool(
            r.get("star6", False) and rn in equip6_map
        )

    # Other toggles / selects
    ename = str(pending.get("enchant", "")).strip()
    if ename in enchants["name"].astype(str).str.strip().tolist():
        st.session_state["ench_select"] = ename
    st.session_state["potions_select"] = pending.get("potions", []) or []
    st.session_state["buffs_select"] = pending.get("buffs", []) or []
    st.session_state["login_bonus_luck"] = int(pending.get("login_bonus_luck", 0))
    st.session_state["meteor_cb"] = bool(pending.get("meteor", False))
    st.session_state["luck_totem_cb"] = bool(pending.get("luck_totem", False))
    st.session_state["str_totem_cb"] = bool(pending.get("str_totem", False))
    st.session_state["anim_cal_planner"] = bool(pending.get("apply_anim", False))


# Info
tab_info, tab_build, tab_opt = st.tabs(
    ["Information", "Build Planner", "Optimizer(BETA)"]
)

with tab_info:
    st.markdown("## What the stats mean")
    st.markdown(
        """
- **Luck** — Base chance driver for rare finds; scaled by Meteor/Luck Totem if toggled.
- **Dig Strength** — Affects how quickly you collect the pan (pairs with **Dig Speed**).
- **Capacity** — Total dirt per pan. More capacity increases yield but may increase time via more **Shakes**.
- **Dig Speed %** — Percentage speed bonus applied to shovel's base speed multiplier.
- **Shake Strength** — Dirt processed per shake (multiplied by Strength Totem).
- **Shake Speed %** — Percentage speed bonus for shaking.
- **Sell Boost** — Multiplier applied to value; used in **Profit rate** only.
- **Size Boost** — Extra block size; included for completeness (does not affect efficiency formula).
- **Modifier** — Generic extra multiplier slot if your data uses it.
    """
    )
    st.markdown("## Terminology")
    st.markdown(
        """
- **Efficiency** is the main metric used to compare builds. It is calculated as: (Luck * Capacity² / (Cycle Time * Animation Time (If animation time is enabled)))
- **Profit** is the effective gain from a build taking into account Efficiency x Sellboost. It is calculated as: Efficiency * (1 + SellBoost / 100)
- **Overhead** is the per-pan constant (picking up, moving, etc.) In the planner you can enable/disable player introduced overhead as if a bot was playing. For more realistic values, use the monte carlo simulation results.
- **Monte Carlo**: Simulates an active use case scenario. Gives the averages for a more realistic distribution of efficiency.
    """
    )
    st.markdown("## How to use the app")
    st.markdown(
        """
1. Select your desired Pan and Shovel from the side menu.
2. Choose your equipment for each slot (Neck, Charm, Rings) using the respective dropdowns.
3. Adjust any relevant settings or toggles in the sidebar.
4. Click the 'Run Optimizer' button to see the best build suggestions.
    """
    )
    st.markdown("## Notes")
    st.markdown(
        """
- This isnt the same value as the CSV everyones using!?! , That csv isnt accurate. This tool properly tracks 6 Star Stats and Pan Passives which have been ignored by the community csv
- The optimizer may not find the absolute best build due to the complexity of the search space.
- Always double-check the suggested builds and adjust based on your playstyle and preferences.
- If you encounter any issues or have suggestions, please reach out to the ItsPure on discord.
- If you have 100% Equipment, 100% 6 Star equipment, etc that isnt in the database. Feel Free to reach out to ItsPure on discord and I will get it added.
- We use a modified version of Nidolya's forumula for calculating efficiency and profit. Instead of a static .625 time multiplier, we use a dynamic multiplier based on the specific equipment and buffs you have active.
    """
    )

# Build Planner
with tab_build:

    def has6(name: str) -> bool:
        return bool(name and name != "(Empty)" and name in equip6_map)

    # Sidebar - Pan / Shovel
    with st.sidebar:
        st.header("Pan & Shovel")
        pan_name = st.selectbox("Pan", pans["name"].tolist(), key="pan_select")
        shv_name = st.selectbox("Shovel", shovels["name"].tolist(), key="shovel_select")

        prow = pans[pans["name"] == pan_name].iloc[0]
        srow = shovels[shovels["name"] == shv_name].iloc[0]
        pan = dict(
            name=prow["name"],
            luck=float(prow["luck"]),
            capacity=float(prow["capacity"]),
            shake_str=float(prow["shake_str"]),
            shake_speed_mult=float(prow["shake_speed_mult"]),
            sell=float(prow.get("sell", 0.0)),
            size=float(prow.get("size", 0.0)),
            modifier=float(prow.get("modifier", 0.0)),
        )

        shovel = dict(
            name=srow["name"],
            dig_str=float(srow["dig_str"]),
            dig_speed_mult=float(srow["dig_speed_mult"]),
            toughness=float(srow["toughness"]),
            sell=float(srow.get("sell", 0.0)),
            size=float(srow.get("size", 0.0)),
            modifier=float(srow.get("modifier", 0.0)),
        )

        st.caption(
            f"**Pan**: {pan['name']} — Luck {int(pan['luck'])}, Cap {int(pan['capacity'])}, Shake STR {pan['shake_str']:.1f}, Time ×{pan['shake_speed_mult']:.2f}"
        )
        st.caption(
            f"**Shovel**: {shovel['name']} — Dig STR {shovel['dig_str']:.1f}, Speed ×{shovel['dig_speed_mult']:.2f} (Tough {int(shovel.get('toughness',0))})"
        )

    # Equipment selection
    left, right = st.columns([1, 2])

    with left:
        st.subheader("Neck")
        neck_df = equip[
            equip["slot"].astype(str).str.strip().str.lower() == "necklace"
        ].copy()
        neck_opts = ["(Empty)"] + neck_df["name"].tolist()
        neck_name = st.selectbox("Neck", neck_opts, key="neck_select", index=0)
        neck_star6 = (
            False
            if neck_name == "(Empty)"
            else st.checkbox(
                "Neck 6★",
                key="neck_star6_cb",
                value=False,
                disabled=not has6(neck_name),
            )
        )

        st.subheader("Charm")
        charm_df = equip[
            equip["slot"].astype(str).str.strip().str.lower() == "charm"
        ].copy()
        charm_opts = ["(Empty)"] + charm_df["name"].tolist()
        charm_name = st.selectbox("Charm", charm_opts, key="charm_select", index=0)
        charm_star6 = (
            False
            if charm_name == "(Empty)"
            else st.checkbox(
                "Charm 6★",
                key="charm_star6_cb",
                value=False,
                disabled=not has6(charm_name),
            )
        )

    with right:
        st.subheader("Rings")
        rings_df = equip[
            equip["slot"].astype(str).str.strip().str.lower() == "ring"
        ].copy()
        ring_opts = ["(Empty)"] + rings_df["name"].tolist()
        colR1, colR2 = st.columns(2)
        ring_choices, ring_star6_flags = [], []
        for i in range(1, 8 + 1):
            col = colR1 if i <= 4 else colR2
            with col:
                choice = st.selectbox(
                    f"Ring {i}", ring_opts, key=f"ring_sel_{i}", index=0
                )
                star6 = False
                if choice != "(Empty)":
                    star6 = st.checkbox(
                        "6★",
                        key=f"ring_star6_{i}",
                        value=False,
                        disabled=not has6(choice),
                    )
                ring_choices.append(choice)
                ring_star6_flags.append(star6)

    # Enchants / Potions / Buffs
    st.subheader("Enchants, Potions & Buffs")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        if "ench_select" in st.session_state:
            ench_name = st.selectbox(
                "Pan Enchant", enchants["name"].tolist(), key="ench_select"
            )
        else:
            # only supply a default index if the key doesn't exist yet
            ench_name = st.selectbox(
                "Pan Enchant", enchants["name"].tolist(), key="ench_select"
            )

    with c2:
        pot_options = potions["name"].astype(str).str.strip().tolist()
        # filter any pre-seeded values to valid options
        if "potions_select" in st.session_state:
            st.session_state["potions_select"] = [
                p
                for p in (st.session_state.get("potions_select") or [])
                if p in pot_options
            ]
            pot_sel = st.multiselect(
                "Potions (stacking)", pot_options, key="potions_select"
            )
        else:
            pot_sel = st.multiselect(
                "Potions (stacking)", pot_options, key="potions_select"
            )

    with c3:
        buff_options = buffs["name"].astype(str).str.strip().tolist()
        if "buffs_select" in st.session_state:
            st.session_state["buffs_select"] = [
                b
                for b in (st.session_state.get("buffs_select") or [])
                if b in buff_options
            ]
            buff_sel = st.multiselect(
                "Permanent Buffs", buff_options, key="buffs_select"
            )
        else:
            buff_sel = st.multiselect(
                "Permanent Buffs", buff_options, key="buffs_select"
            )

    with c4:
        login_bonus_luck = st.number_input(
            "Extra Luck from login days",
            key="login_bonus_luck",
            value=0,
            step=1,
            min_value=0,
        )

    # Totems / Animation
    st.subheader("Run Toggles")
    t1, t2, t3, t4 = st.columns(4)
    with t1:
        meteor = st.checkbox("Meteor active (Luck ×2)", key="meteor_cb", value=False)
    with t2:
        luck_totem = st.checkbox(
            "Luck Totem (Luck ×2)", key="luck_totem_cb", value=False
        )
    with t3:
        str_totem = st.checkbox(
            "Strength Totem (STR ×2)", key="str_totem_cb", value=False
        )
    with t4:
        apply_anim = st.checkbox(
            "Apply animation time (+50%)", value=False, key="anim_cal_planner"
        )
    anim_mult = (1.0 + ANIM_EXTRA) if apply_anim else 1.0

    luck_mult = (2.0 if meteor else 1.0) * (2.0 if luck_totem else 1.0)
    str_mult = 2.0 if str_totem else 1.0

    # Build item list
    items = []
    if neck_name != "(Empty)":
        nrow = neck_df[neck_df["name"] == neck_name]
        if not nrow.empty:
            items.append(
                row_to_item(
                    nrow.iloc[0], neck_star6, overrides=equip6_map.get(neck_name)
                )
            )
    if charm_name != "(Empty)":
        crow = charm_df[charm_df["name"] == charm_name]
        if not crow.empty:
            items.append(
                row_to_item(
                    crow.iloc[0], charm_star6, overrides=equip6_map.get(charm_name)
                )
            )
    for choice, is6 in zip(ring_choices, ring_star6_flags):
        if choice == "(Empty)":
            continue
        rrow = rings_df[rings_df["name"] == choice]
        if not rrow.empty:
            items.append(
                row_to_item(rrow.iloc[0], is6, overrides=equip6_map.get(choice))
            )

    # Resolve rows
    ench_row = (
        enchants[enchants["name"] == ench_name].iloc[0]
        if len(enchants[enchants["name"] == ench_name])
        else None
    )
    potion_rows = [
        potions[potions["name"] == n].iloc[0]
        for n in pot_sel
        if n in potions["name"].tolist()
    ]
    buff_rows = [
        buffs[buffs["name"] == n].iloc[0]
        for n in buff_sel
        if n in buffs["name"].tolist()
    ]

    # Compute / show results — Overhead in planner is ideal cycle only (0.0 human delay)
    totals, res = compute_once(
        pan,
        dict(shovel),
        items,
        ench_row,
        potion_rows,
        buff_rows,
        luck_mult,
        str_mult,
        0.0,
        login_bonus_luck,
        anim_mult=anim_mult,
    )

    base_dig_pct = 100.0 + (float(shovel.get("dig_speed_mult", 1.0)) - 1.0) * 100.0
    base_shake_pct = 100.0 + (float(pan.get("shake_speed_mult", 1.0)) - 1.0) * 100.0

    dig_speed_ui = totals.get("dig_speed", 0.0) + base_dig_pct
    shake_speed_ui = totals.get("shake_speed", 0.0) + base_shake_pct

    st.markdown(
        '<h2 style="font-size:2.0rem;margin:0.25rem 0 0.75rem 0">Results</h2>',
        unsafe_allow_html=True,
    )
    st.subheader("Build Stats (Totals)")
    stats_cols = st.columns(5)
    with stats_cols[0]:
        st.write(f"**Luck:** {int(totals['luck'])}")
        st.write(f"**Dig Strength:** {totals['dig_str']:.1f}")
    with stats_cols[1]:
        st.write(f"**Capacity:** {int(totals['capacity'])}")
        st.write(f"**Dig Speed %:** {dig_speed_ui:.1f}%")
    with stats_cols[2]:
        st.write(f"**Shake Strength:** {totals['shake_str']:.1f}")
        st.write(f"**Shake Speed %:** {shake_speed_ui:.1f}%")
    with stats_cols[3]:
        st.write(f"**Sell Boost:** {totals['sell']:.1f}%")
        st.write(f"**Size Boost:** {totals['size']:.1f}%")
    with stats_cols[4]:
        st.write(f"**Modifier Boost:** {totals['modifier']:.1f}%")
        st.write(f"**Toughness:** {shovel['toughness']:}")
    st.subheader("Derived (from build)")
    d1, d2 = st.columns(2)
    with d1:
        st.write(f"**Luck (effective):** {int(res['eff_luck']):,}")
        st.write(f"**Shake STR (effective):** {res['eff_shake_str']:.2f}")
    with d2:
        st.write(f"**Shakes:** {res['shakes']}")
        st.write(f"**Digs:** {res['digs']}")
        st.write(f"**Cycle time (ideal):** {res['time_per_pan_s']:.3f} s")

    st.subheader("Key metrics")
    k1, k2 = st.columns(2)
    with k1:
        st.markdown(f"### Efficiency: {int(res['efficiency']):,}")
        st.caption(
            "Efficiency = (Luck_eff × √Capacity) / (Cycle Time × Animation Multiplier)"
        )
    with k2:
        st.markdown(f"### Profit rate: {int(res['profit_rate']):,}")
        st.caption("Profit rate = Efficiency × (1 + Sell/100)")

    # MC
    st.divider()
    st.markdown("#### Monte Carlo (optional)")
    mc_toggle = st.checkbox(
        "Run Monte Carlo for realistic delay", key="mc_planner_toggle"
    )
    if mc_toggle:
        c1, c2, c3 = st.columns(3)
        with c1:
            mu = st.number_input(
                "Average extra delay (s)", value=0.8, step=0.1, key="mc_mu_planner"
            )
        with c2:
            sigma = st.number_input(
                "Delay variability, std dev (s)",
                value=0.3,
                step=0.05,
                key="mc_sigma_planner",
            )
        with c3:
            runs = st.number_input(
                "Runs", value=400, step=50, min_value=50, key="mc_runs_planner"
            )
        c4, c5 = st.columns(2)
        with c4:
            omin = st.number_input(
                "Min delay clamp (s)", value=0.0, step=0.1, key="mc_min_planner"
            )
        with c5:
            omax = st.number_input(
                "Max delay clamp (s)", value=2.0, step=0.1, key="mc_max_planner"
            )

        summary = mc_overhead_sim(
            pan,
            dict(shovel),
            items,
            ench_row,
            potion_rows,
            buff_rows,
            luck_mult,
            str_mult,
            float(mu),
            float(sigma),
            float(omin),
            float(omax),
            login_bonus_luck,
            runs=int(runs),
            anim_mult=anim_mult,
        )
        st.write(
            f"**Typical Cycle:** {summary['time_mean']:.3f} s  (spread ±{summary['time_std']:.3f})"
        )
        st.write(
            f"**Typical Efficiency:** {summary['eff_mean']:.0f}  (spread ±{summary['eff_std']:.0f})"
        )
        st.write(
            f"**Typical Profit:** {summary['prof_mean']:.0f}  (spread ±{summary['prof_std']:.0f})"
        )

    # Selected
    st.divider()
    st.markdown("#### Selected Items")
    _df_items = pd.DataFrame(items)
    _cols = [
        "name",
        "star6",
        "luck",
        "dig_str",
        "capacity",
        "dig_speed",
        "shake_str",
        "shake_speed",
        "sell",
        "size",
        "modifier",
    ]
    if _df_items.empty:
        _df_items = pd.DataFrame(columns=_cols)
    else:
        _df_items = _df_items.reindex(columns=_cols, fill_value=0)
    st.dataframe(_df_items, use_container_width=True)

    # Export/Import
    st.divider()
    st.markdown("### Presets — Save / Load")
    current_build = {
        "pan_name": pan["name"],
        "shovel_name": shovel["name"],
        "neck": {"name": neck_name, "star6": bool(neck_star6)},
        "charm": {"name": charm_name, "star6": bool(charm_star6)},
        "rings": [
            {"name": ring_choices[i], "star6": bool(ring_star6_flags[i])}
            for i in range(len(ring_choices))
        ],
        "enchant": ench_name,
        "potions": pot_sel,
        "buffs": buff_sel,
        "login_bonus_luck": int(login_bonus_luck),
        "meteor": bool(meteor),
        "luck_totem": bool(luck_totem),
        "str_totem": bool(str_totem),
        "apply_anim": bool(apply_anim),
    }

    colP1, colP2 = st.columns(2)
    with colP1:
        st.markdown("#### Export current build")
        export_json = json.dumps(current_build, indent=2)
        with st.expander("Preview export JSON", expanded=False):
            st.code(export_json, language="json")
        st.download_button(
            "Download build JSON",
            export_json,
            file_name="prospecting_build.json",
            mime="application/json",
        )

    with colP2:
        st.markdown("#### Import build (paste JSON)")
        with st.form("import_form_bottom", clear_on_submit=False):
            pasted = st.text_area(
                "Paste build JSON here",
                key="import_text_bottom",
                height=160,
                placeholder="{ ... }",
            )
            submitted = st.form_submit_button("Load pasted build")
        if submitted:
            try:
                payload = json.loads(pasted) if pasted and pasted.strip() else None
                if payload:
                    st.session_state["pending_import_payload"] = payload
                    st.success("Build imported. Applying controls…")
                    st.rerun()
                else:
                    st.warning("No JSON found.")
            except Exception as e:
                st.error(f"Failed to parse JSON: {e}")

# Optimizer
with tab_opt:
    st.subheader("Optimize / Generate Build(BETA)")

    # Pools
    neck_df_all = equip[
        equip["slot"].astype(str).str.strip().str.lower() == "necklace"
    ].copy()
    charm_df_all = equip[
        equip["slot"].astype(str).str.strip().str.lower() == "charm"
    ].copy()
    ring_df_all = equip[
        equip["slot"].astype(str).str.strip().str.lower() == "ring"
    ].copy()

    all_pans = pans["name"].tolist()
    all_shovels = shovels["name"].tolist()
    all_enchants = enchants["name"].tolist()

    # Budgets / search
    b1, b2, b3 = st.columns(3)
    with b1:
        max_combos = st.number_input(
            "Max (Pan×Shovel×Enchant) to explore",
            value=100,
            min_value=1,
            step=50,
            key="opt_max_combos",
        )
    with b2:
        neck_charm_shortlist = st.number_input(
            "Neck/Charm shortlist per combo",
            value=6,
            min_value=1,
            step=1,
            key="opt_nc_short",
        )
    with b3:
        ring_shortlist = st.number_input(
            "Ring shortlist size", value=30, min_value=5, step=5, key="opt_ring_short"
        )

    b4, b5, b6 = st.columns(3)
    with b4:
        search_runs = st.number_input(
            "Greedy builds per shortlist",
            value=12,
            min_value=1,
            step=1,
            key="opt_search_runs",
        )
    with b5:
        jitter_pct = st.number_input(
            "Greedy jitter (±%)", value=1.0, min_value=0.0, step=0.5, key="opt_jitter"
        )
    with b6:
        rand_seed = st.number_input(
            "Random seed", value=42, min_value=0, step=1, key="opt_seed"
        )

    # Objective / constraints
    ctop1, ctop2, ctop3 = st.columns(3)
    with ctop1:
        objective = st.selectbox(
            "Objective",
            ["Mythic/Exotics rate (Efficiency)", "Making Money (Profit rate)"],
            key="opt_objective",
        )
    with ctop2:
        allow_dupes = st.checkbox("Allow duplicate rings", value=True, key="opt_dupes")
    with ctop3:
        include_empty = st.checkbox(
            "Allow empty ring slots", value=False, key="opt_empty"
        )

    st.markdown("### Whitelists & Limits")
    # Pans/Shovels — single choice
    ps1, ps2 = st.columns(2)
    with ps1:
        pan_choice = st.selectbox(
            "Pan (exactly one)",
            options=all_pans,
            index=(
                all_pans.index(st.session_state.get("pan_select", all_pans[0]))
                if all_pans
                else 0
            ),
            key="wl_pan_choice",
        )
    with ps2:
        shv_choice = st.selectbox(
            "Shovel (exactly one)",
            options=all_shovels,
            index=(
                all_shovels.index(st.session_state.get("shovel_select", all_shovels[0]))
                if all_shovels
                else 0
            ),
            key="wl_shovel_choice",
        )

    # Enchants — unlimited multiselect
    ench_wl = st.multiselect(
        "Enchants (no limit)",
        options=all_enchants,
        default=all_enchants,
        key="wl_enchants_fast",
    )

    # Helper to enforce max selections for the gear categories
    def enforce_max4(lst, label):
        if len(lst) > 4:
            st.info(f"{label}: using first 4 of {len(lst)} selections.")
            return lst[:4]
        return lst

    with st.expander("Necklaces (max 4 used)", expanded=False):
        neck_wl = st.multiselect(
            "Use these necklaces",
            options=neck_df_all["name"].tolist(),
            default=neck_df_all["name"].tolist(),
            key="wl_necks_fast",
        )
        neck_wl = enforce_max4(neck_wl, "Necklaces")

    with st.expander("Charms (max 4 used)", expanded=False):
        charm_wl = st.multiselect(
            "Use these charms",
            options=charm_df_all["name"].tolist(),
            default=charm_df_all["name"].tolist(),
            key="wl_charms_fast",
        )
        charm_wl = enforce_max4(charm_wl, "Charms")

    with st.expander("Rings (max 4 used)", expanded=False):
        ring_wl = st.multiselect(
            "Use these rings",
            options=ring_df_all["name"].tolist(),
            default=ring_df_all["name"].tolist(),
            key="wl_rings_fast",
        )
        ring_wl = enforce_max4(ring_wl, "Rings")

    # Toggles / Animation / MC verification
    t1, t2, t3, t4 = st.columns(4)
    with t1:
        meteor = st.checkbox(
            "Meteor active (Luck ×2)",
            value=bool(st.session_state.get("meteor_cb", False)),
            key="meteor_fast",
        )
    with t2:
        luck_totem = st.checkbox(
            "Luck Totem (Luck ×2)",
            value=bool(st.session_state.get("luck_totem_cb", False)),
            key="luck_fast",
        )
    with t3:
        str_totem = st.checkbox(
            "Strength Totem (STR ×2)",
            value=bool(st.session_state.get("str_totem_cb", False)),
            key="str_fast",
        )
    with t4:
        apply_anim_opt = st.checkbox(
            "Apply animation time in scoring (+50%)",
            value=bool(st.session_state.get("anim_cal_planner", False)),
            key="anim_cal_opt",
        )
    anim_mult_opt = (1.0 + ANIM_EXTRA) if apply_anim_opt else 1.0

    # 6 star usage in optimizer
    allow_six = st.checkbox("Enable 6★ where available", value=True, key="opt_allow6")

    luck_mult = (2.0 if meteor else 1.0) * (2.0 if luck_totem else 1.0)
    str_mult = 2.0 if str_totem else 1.0
    login_bonus_luck = st.number_input(
        "Extra Luck from login days",
        value=int(st.session_state.get("login_bonus_luck", 0)),
        step=1,
        min_value=0,
        key="opt_login_b",
    )

    mc1, mc2, mc3 = st.columns(3)
    with mc1:
        mc_runs = st.number_input(
            "MC runs per candidate", value=300, min_value=50, step=50, key="opt_mc_runs"
        )
    with mc2:
        mc_sigma = st.number_input(
            "Overhead std dev (s)", value=0.3, step=0.05, key="opt_mc_sigma"
        )
    with mc3:
        mc_range = st.slider(
            "Overhead clamp range (±s)", 0.0, 2.0, 1.0, 0.1, key="opt_mc_range"
        )

    # Average overhead (mean) for optimizer scoring & MC, clamps symmetrical around it
    overhead_s = st.number_input(
        "Average overhead (s) used in MC", value=0.8, step=0.1, key="opt_overhead_mu"
    )
    overhead_min = max(0.0, overhead_s - mc_range)
    overhead_max = overhead_s + mc_range

    # Helper shortlist functions
    def shortlist_neck_charm(pan, shovel, ench_row, neck_df, charm_df, k, anim_mult):
        scores = []
        for _, n in neck_df.iterrows():
            n_item = row_to_item(
                n, allow_six and (n["name"] in equip6_map), equip6_map.get(n["name"])
            )
            for _, c in charm_df.iterrows():
                c_item = row_to_item(
                    c,
                    allow_six and (c["name"] in equip6_map),
                    equip6_map.get(c["name"]),
                )
                items = [n_item, c_item]
                _, res = compute_once(
                    pan,
                    shovel,
                    items,
                    ench_row,
                    [],
                    [],
                    luck_mult,
                    str_mult,
                    0.0,
                    login_bonus_luck,
                    anim_mult=anim_mult,
                )
                scores.append(((n_item, c_item), score_from_res(res, objective)))
        scores.sort(key=lambda x: x[1], reverse=True)
        return [pair for pair, _ in scores[: int(k)]]

    def shortlist_rings(
        pan,
        shovel,
        ench_row,
        base_items,
        ring_df,
        size,
        allow_dupes,
        include_empty,
        anim_mult,
    ):
        candidates = []
        if include_empty:
            candidates.append(empty_item("(Empty)"))
        lone_scores = []
        for _, r in ring_df.iterrows():
            ring = row_to_item(
                r, allow_six and (r["name"] in equip6_map), equip6_map.get(r["name"])
            )
            test = base_items + [ring]
            _, res = compute_once(
                pan,
                shovel,
                test,
                ench_row,
                [],
                [],
                luck_mult,
                str_mult,
                0.0,
                login_bonus_luck,
                anim_mult=anim_mult,
            )
            lone_scores.append((ring, score_from_res(res, objective)))
        lone_scores.sort(key=lambda x: x[1], reverse=True)
        for ring, _ in lone_scores[: int(size)]:
            candidates.append(ring)
        return candidates

    def greedy_fill(
        pan,
        shovel,
        ench_row,
        base_items,
        ring_candidates,
        runs,
        jitter_pct,
        allow_dupes,
        seed,
        anim_mult,
    ):
        rng = np.random.default_rng(seed)
        best_build, best_score = None, -1e99
        for _ in range(int(runs)):
            chosen, used = [], set()
            for slot in range(8):
                local_best, local_best_score = None, -1e99
                for cand in ring_candidates:
                    if (
                        cand["name"] != "(Empty)"
                        and (not allow_dupes)
                        and cand["name"] in used
                    ):
                        continue
                    test = (
                        base_items
                        + chosen
                        + ([] if cand["name"] == "(Empty)" else [cand])
                    )
                    _, res = compute_once(
                        pan,
                        shovel,
                        test,
                        ench_row,
                        [],
                        [],
                        luck_mult,
                        str_mult,
                        0.0,
                        login_bonus_luck,
                        anim_mult=anim_mult,
                    )
                    s = score_from_res(res, objective) * (
                        1.0 + rng.normal(0, float(jitter_pct) / 100.0)
                    )
                    if s > local_best_score:
                        local_best_score, local_best = s, cand
                if local_best and local_best["name"] != "(Empty)":
                    chosen.append(local_best)
                    used.add(local_best["name"])
            _, final_res = compute_once(
                pan,
                shovel,
                base_items + chosen,
                ench_row,
                [],
                [],
                luck_mult,
                str_mult,
                0.0,
                login_bonus_luck,
                anim_mult=anim_mult,
            )
            final_score = score_from_res(final_res, objective)
            if final_score > best_score:
                best_score, best_build = final_score, dict(
                    items=(base_items + chosen), res=final_res
                )
        return best_build

    # Button to run optimizer
    if st.button("Run Optimizer", key="opt_run_btn"):
        with st.spinner("Searching…"):
            # Filter by chosen pan/shovel and limited whitelists
            neck_df = neck_df_all[neck_df_all["name"].isin(neck_wl)].copy()
            charm_df = charm_df_all[charm_df_all["name"].isin(charm_wl)].copy()
            ring_df = ring_df_all[ring_df_all["name"].isin(ring_wl)].copy()

            ench_df = enchants[enchants["name"].isin(ench_wl)].copy()
            if ench_df.empty:
                st.error("Please select at least one Enchant.")
                st.stop()

            # Build combos across chosen pan/shovel and selected enchants only
            combos = [(pan_choice, shv_choice, e) for e in ench_df["name"].tolist()]
            combos = combos[: int(max_combos)]

            progress = st.progress(0, text="Initializing…")
            builds = []
            total_steps = max(1, len(combos))

            for idx, (pname, sname, ename) in enumerate(combos, start=1):
                progress.progress(
                    min(1.0, idx / total_steps),
                    text=f"Combo {idx}/{total_steps}: {pname} × {sname} × {ename}",
                )

                # --- Safe row lookups (trimmed names) ---
                _pname = str(pname).strip()
                _sname = str(sname).strip()
                _ename = str(ename).strip()

                prow_df = pans.loc[pans["name"].astype(str).str.strip() == _pname]
                if prow_df.empty:
                    st.warning(f"Pan '{_pname}' not found; skipping.")
                    continue
                srow_df = shovels.loc[shovels["name"].astype(str).str.strip() == _sname]
                if srow_df.empty:
                    st.warning(f"Shovel '{_sname}' not found; skipping.")
                    continue
                erow_df = enchants.loc[
                    enchants["name"].astype(str).str.strip() == _ename
                ]
                if erow_df.empty:
                    st.warning(f"Enchant '{_ename}' not found; skipping.")
                    continue

                prow = prow_df.iloc[0]
                srow = srow_df.iloc[0]
                ench_row = erow_df.iloc[0]

                # --- Build the pan/shovel objects for THIS combo (don't use planner's pan/shovel) ---
                pan_obj = dict(
                    name=prow["name"],
                    luck=float(prow["luck"]),
                    capacity=float(prow["capacity"]),
                    shake_str=float(prow["shake_str"]),
                    shake_speed_mult=float(prow["shake_speed_mult"]),
                    sell=float(prow.get("sell", 0.0)),
                    size=float(prow.get("size", 0.0)),
                    modifier=float(prow.get("modifier", 0.0)),
                )
                shovel_obj = dict(
                    name=srow["name"],
                    dig_str=float(srow["dig_str"]),
                    dig_speed_mult=float(srow["dig_speed_mult"]),
                    toughness=float(srow.get("toughness", 0.0)),
                    sell=float(srow.get("sell", 0.0)),
                    size=float(srow.get("size", 0.0)),
                    modifier=float(srow.get("modifier", 0.0)),
                )

                # --- shortlist + greedy using the combo's pan/shovel ---
                nc_pairs = shortlist_neck_charm(
                    pan_obj,
                    shovel_obj,
                    ench_row,
                    neck_df,
                    charm_df,
                    k=int(neck_charm_shortlist),
                    anim_mult=anim_mult_opt,
                )

                for n_item, c_item in nc_pairs:
                    base_items = [n_item, c_item]
                    ring_cands = shortlist_rings(
                        pan_obj,
                        shovel_obj,
                        ench_row,
                        base_items,
                        ring_df,
                        size=int(ring_shortlist),
                        allow_dupes=allow_dupes,
                        include_empty=include_empty,
                        anim_mult=anim_mult_opt,
                    )
                    best = greedy_fill(
                        pan_obj,
                        shovel_obj,
                        ench_row,
                        base_items,
                        ring_cands,
                        runs=int(search_runs),
                        jitter_pct=jitter_pct,
                        allow_dupes=allow_dupes,
                        seed=int(rand_seed),
                        anim_mult=anim_mult_opt,
                    )
                    if best:
                        item_names = [i["name"] for i in best["items"]]
                        builds.append(
                            dict(
                                pan_name=_pname,
                                shovel_name=_sname,
                                ench_name=_ename,
                                neck=n_item["name"],
                                charm=c_item["name"],
                                items=best["items"],
                                res=best["res"],
                                item_names=item_names,
                            )
                        )

            progress.progress(1.0, text="Ranking results…")

            if not builds:
                st.error(
                    "No candidates found. Try increasing budgets/shortlists or whitelisting more gear."
                )
            else:
                # Rank by objective
                builds.sort(
                    key=lambda d: score_from_res(d["res"], objective), reverse=True
                )
                top_show = min(15, len(builds))
                st.markdown("#### Top deterministic candidates")
                rows = []
                for i in range(top_show):
                    b = builds[i]
                    res = b["res"]
                    rows.append(
                        dict(
                            Rank=i + 1,
                            Pan=b["pan_name"],
                            Shovel=b["shovel_name"],
                            Enchant=b["ench_name"],
                            Neck=b["neck"],
                            Charm=b["charm"],
                            Efficiency=int(res["efficiency"]),
                            Profit=int(res["profit_rate"]),
                            Shakes=res["shakes"],
                            Time=f"{res['time_per_pan_s']:.3f}s",
                        )
                    )
                st.dataframe(pd.DataFrame(rows), use_container_width=True)

                # MC verify top N
                st.markdown("#### Monte Carlo verification")
                top_k = min(8, len(builds))
                ver_rows = []
                for i in range(top_k):
                    b = builds[i]
                    mc = mc_overhead_sim(
                        dict(
                            name=b["pan_name"],
                            luck=float(
                                pans[pans["name"] == b["pan_name"]].iloc[0]["luck"]
                            ),
                            capacity=float(
                                pans[pans["name"] == b["pan_name"]].iloc[0]["capacity"]
                            ),
                            shake_str=float(
                                pans[pans["name"] == b["pan_name"]].iloc[0]["shake_str"]
                            ),
                            shake_speed_mult=float(
                                pans[pans["name"] == b["pan_name"]].iloc[0][
                                    "shake_speed_mult"
                                ]
                            ),
                        ),
                        dict(
                            name=b["shovel_name"],
                            dig_str=float(
                                shovels[shovels["name"] == b["shovel_name"]].iloc[0][
                                    "dig_str"
                                ]
                            ),
                            dig_speed_mult=float(
                                shovels[shovels["name"] == b["shovel_name"]].iloc[0][
                                    "dig_speed_mult"
                                ]
                            ),
                            toughness=float(
                                shovels[shovels["name"] == b["shovel_name"]].iloc[0][
                                    "toughness"
                                ]
                            ),
                        ),
                        b["items"],
                        enchants[enchants["name"] == b["ench_name"]].iloc[0],
                        [],
                        [],
                        luck_mult,
                        str_mult,
                        float(overhead_s),
                        float(mc_sigma),
                        float(overhead_min),
                        float(overhead_max),
                        int(login_bonus_luck),
                        runs=int(mc_runs),
                        seed=int(rand_seed) + i,
                        anim_mult=anim_mult_opt,
                    )
                    b["mc_eff_mean"] = mc["eff_mean"]
                    b["mc_eff_std"] = mc["eff_std"]
                    b["mc_prof_mean"] = mc["prof_mean"]
                    b["mc_prof_std"] = mc["prof_std"]
                    b["mc_time_mean"] = mc["time_mean"]
                    b["mc_time_std"] = mc["time_std"]
                    ver_rows.append(
                        dict(
                            Pan=b["pan_name"],
                            Shovel=b["shovel_name"],
                            Enchant=b["ench_name"],
                            Neck=b["neck"],
                            Charm=b["charm"],
                            Typical_Efficiency=int(mc["eff_mean"]),
                            Efficiency_spread=f"±{mc['eff_std']:.1f}",
                            Typical_Profit=int(mc["prof_mean"]),
                            Profit_spread=f"±{mc['prof_std']:.1f}",
                            Typical_Cycle=f"{mc['time_mean']:.3f}s",
                            Cycle_spread=f"±{mc['time_std']:.3f}s",
                        )
                    )
                st.dataframe(pd.DataFrame(ver_rows), use_container_width=True)

                # Choose MC best
                if objective.startswith("Mythic"):
                    best = max(builds[:top_k], key=lambda b: b["mc_eff_mean"])
                else:
                    best = max(builds[:top_k], key=lambda b: b["mc_prof_mean"])

                # Build an export payload for Planner
                neck_name = best["neck"]
                charm_name = best["charm"]
                rings_struct = [
                    {"name": nm, "star6": (allow_six and (nm in equip6_map))}
                    for nm in best["item_names"]
                    if nm not in (neck_name, charm_name)
                ]
                while len(rings_struct) < 8:
                    rings_struct.append({"name": "(Empty)", "star6": False})

                best_build = {
                    "pan_name": best["pan_name"],
                    "shovel_name": best["shovel_name"],
                    "neck": {
                        "name": neck_name,
                        "star6": (allow_six and (neck_name in equip6_map)),
                    },
                    "charm": {
                        "name": charm_name,
                        "star6": (allow_six and (charm_name in equip6_map)),
                    },
                    "rings": rings_struct,
                    "enchant": best["ench_name"],
                    "potions": [],
                    "buffs": [],
                    "login_bonus_luck": int(login_bonus_luck),
                    "meteor": bool(meteor),
                    "luck_totem": bool(luck_totem),
                    "str_totem": bool(str_totem),
                    "apply_anim": bool(apply_anim_opt),
                }

                st.success(
                    f"MC-best by {'Efficiency' if objective.startswith('Mythic') else 'Profit'} → Pan {best['pan_name']} | Shovel {best['shovel_name']} | Enchant {best['ench_name']}"
                )

                c1, c2 = st.columns(2)
                with c1:
                    if st.button(
                        "Apply best build to Planner (Broken ATM)",
                        key="apply_best_to_planner",
                    ):
                        st.session_state["pending_import_payload"] = best_build
                        st.rerun()
                with c2:
                    st.download_button(
                        "Download best build JSON",
                        json.dumps(best_build, indent=2),
                        file_name="best_build.json",
                        mime="application/json",
                    )
