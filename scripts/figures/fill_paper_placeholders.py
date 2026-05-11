"""
Fill the \\PH... placeholder macros in report/arxiv/preamble.tex with
actual numbers from the experiment JSON outputs.

Reads:
  report/arxiv/data/multi_window_results.json   (W1, W2 MAPE + CIs)
  report/arxiv/data/btm_correlation.json        (Spearman + p-value)
  report/arxiv/data/future_weather_shift.json   (forecast vs analyses delta)

Writes back to report/arxiv/preamble.tex by re-defining the
\\PHWONEBASELINE / \\PHWTWOBASELINE / etc. macros to expand to formatted
text (e.g., "25.17 % [22.34--28.20]") instead of the placeholder TBD.

Run AFTER Phase D + Phase G (BTM correlation) + Phase E (forecast shift)
all produce their JSON outputs; before the final latexmk build.

Idempotent — running twice produces the same output.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PREAMBLE = ROOT / "report" / "arxiv" / "preamble.tex"
TBD = "\\textcolor{BadRed}{TBD}"

MULTI_WINDOW = ROOT / "report" / "arxiv" / "data" / "multi_window_results.json"
BTM = ROOT / "report" / "arxiv" / "data" / "btm_correlation.json"
SHIFT = ROOT / "report" / "arxiv" / "data" / "future_weather_shift.json"
MULTI_YEAR = ROOT / "report" / "arxiv" / "data" / "multi_year_drift.json"
DW_BENCH = ROOT / "report" / "arxiv" / "data" / "drift_weighted_benchmark.json"
HOUR_OF_DAY = ROOT / "report" / "arxiv" / "data" / "hour_of_day_mape.json"
LOAD_CURVES = ROOT / "report" / "arxiv" / "data" / "load_curves_summary.json"


def fmt_ci(point: float, ci_low: float, ci_high: float) -> str:
    return f"{point:.2f}~\\% [{ci_low:.2f}--{ci_high:.2f}]"


def replace_macro(text: str, macro: str, body: str) -> str:
    """Replace `\\newcommand{\\<macro>}{...}` with body. Uses a callable
    repl so backslashes in `body` aren't reinterpreted as group refs."""
    pattern = re.compile(
        r"\\newcommand\{\\" + re.escape(macro) + r"\}\{[^{}]*\}",
        re.MULTILINE)
    new_def = f"\\newcommand{{\\{macro}}}{{{body}}}"
    if pattern.search(text):
        return pattern.sub(lambda m: new_def, text)
    print(f"  WARN: \\{macro} macro not found in preamble")
    return text


def main():
    text = PREAMBLE.read_text()

    if MULTI_WINDOW.exists():
        d = json.loads(MULTI_WINDOW.read_text())
        windows = {w["label"]: w for w in d["windows"]}
        if "W1" in windows and "W2" in windows:
            for label, prefix in (("W1", "WONE"), ("W2", "WTWO")):
                s = windows[label]["summary"]
                for model, suffix in (("baseline", "BASELINE"),
                                       ("chronos", "CHRONOS"),
                                       ("ensemble", "ENSEMBLE"),
                                       ("persistence_1d", "PERSI"),
                                       ("persistence_7d", "PERS"),
                                       ("climatological", "CLIM")):
                    o = s[model]["overall"]
                    body = fmt_ci(o["point"], o["ci_low"], o["ci_high"])
                    text = replace_macro(text, f"PH{prefix}{suffix}", body)

            # baseline drift between windows
            w1_b = windows["W1"]["summary"]["baseline"]["overall"]["point"]
            w2_b = windows["W2"]["summary"]["baseline"]["overall"]["point"]
            text = replace_macro(text, "PHDIFFBASELINE",
                                  f"+{w2_b - w1_b:.1f}")

            # RI/SEMA/WCMA average extreme
            for label, prefix in (("W1", "ONE"), ("W2", "MEMETWO")):
                pz = windows[label]["summary"]["baseline"]["per_zone"]
                avg = (pz["RI"]["point"] + pz["SEMA"]["point"] + pz["WCMA"]["point"]) / 3
                text = replace_macro(text, f"PHRISEMAE{prefix}", f"{avg:.0f}~\\%")
        else:
            print("  WARN: multi_window_results.json missing W1 or W2")

    if BTM.exists():
        b = json.loads(BTM.read_text())
        text = replace_macro(text, "PHRHO", f"{b['spearman_rho']:+.2f}")
        text = replace_macro(text, "PHPVAL", f"{b['permutation_p_two_sided']:.3f}")
        if "permutation_p_one_sided" in b:
            text = replace_macro(text, "PHPVALONE",
                                  f"{b['permutation_p_one_sided']:.3f}")

    if SHIFT.exists():
        s = json.loads(SHIFT.read_text())
        agg = s["aggregate"]["diff_forecast_minus_analyses_pp"]
        text = replace_macro(text, "PHFCSHIFTPP", f"{agg['point']:+.2f}")
        text = replace_macro(text, "PHFCSHIFTLO", f"{agg['ci_low']:+.2f}")
        text = replace_macro(text, "PHFCSHIFTHI", f"{agg['ci_high']:+.2f}")
        # date with largest absolute diff
        per_date = s["per_date"]
        max_d = max(per_date, key=lambda d: abs(d["diff_pp"]))
        text = replace_macro(text, "PHFCMAXDATE", max_d["date"][:10])

    # Build appendix F tables
    if MULTI_WINDOW.exists():
        d = json.loads(MULTI_WINDOW.read_text())
        windows = {w["label"]: w for w in d["windows"]}
        for label, prefix in (("W1", "WONE"), ("W2", "WTWO")):
            if label not in windows:
                continue
            s = windows[label]["summary"]
            zones = ["ME", "NH", "VT", "CT", "RI", "SEMA", "WCMA", "NEMA_BOST"]
            lines = ["\\begin{tabular}{@{}lrrrrrrrr@{}}",
                     "\\toprule",
                     " & " + " & ".join(z.replace("_", "-") for z in zones) + " \\\\",
                     "\\midrule"]
            for model in ("baseline", "chronos", "ensemble",
                          "persistence_1d", "persistence_7d", "climatological"):
                row = [model.replace("_", "-")]
                for z in zones:
                    pz = s[model]["per_zone"][z]
                    row.append(f"{pz['point']:.1f}")
                lines.append(" & ".join(row) + " \\\\")
            lines += ["\\bottomrule", "\\end{tabular}"]
            body = "\n".join(lines)
            # Need a different placeholder representation - use \input strategy
            # For now write the table to a separate .tex file the appendix \input s
            tbl_path = ROOT / "report" / "arxiv" / f"_tbl_{label.lower()}.tex"
            tbl_path.write_text(body + "\n")
            text = replace_macro(text, f"PHTABLE{prefix}",
                                  f"\\input{{_tbl_{label.lower()}}}")

    # NEW v1.6: hour-of-day MAPE headlines
    if HOUR_OF_DAY.exists():
        h = json.loads(HOUR_OF_DAY.read_text())
        r = h["summary"]["results"]["W2"]["baseline"]
        text = replace_macro(text, "PHMIDDAYBASEW",
                              f"{r['midday_high_btm_mape_pct']:.1f}~\\%")
        text = replace_macro(text, "PHNONMIDBASEW",
                              f"{r['nonmidday_high_btm_mape_pct']:.1f}~\\%")
        ratio = r['midday_high_btm_mape_pct'] / r['nonmidday_high_btm_mape_pct']
        text = replace_macro(text, "PHMIDDAYRATIOBASE", f"{ratio:.1f}$\\times$")
        c = h["summary"]["results"]["W2"]["chronos"]
        text = replace_macro(text, "PHMIDDAYCHRONOSW",
                              f"{c['midday_high_btm_mape_pct']:.1f}~\\%")
        text = replace_macro(text, "PHNONMIDCHRONOSW",
                              f"{c['nonmidday_high_btm_mape_pct']:.1f}~\\%")
        ratio_c = c['midday_high_btm_mape_pct'] / c['nonmidday_high_btm_mape_pct']
        text = replace_macro(text, "PHMIDDAYRATIOCHRONOS",
                              f"{ratio_c:.1f}$\\times$")

    # NEW v1.6: duck-curve depth headlines
    if LOAD_CURVES.exists():
        l = json.loads(LOAD_CURVES.read_text())
        d = l["duck_curve_depth"]
        text = replace_macro(text, "PHDUCKWCMAONE",
                              f"{d['W1_WCMA']['duck_curve_depth_pct']:.1f}~\\%")
        text = replace_macro(text, "PHDUCKWCMATWO",
                              f"{d['W2_WCMA']['duck_curve_depth_pct']:.1f}~\\%")
        shift_wcma = (d['W2_WCMA']['duck_curve_depth_pct']
                       - d['W1_WCMA']['duck_curve_depth_pct'])
        text = replace_macro(text, "PHDUCKWCMASHIFT", f"+{shift_wcma:.1f}~pp")
        text = replace_macro(text, "PHDUCKMEONE",
                              f"{d['W1_ME']['duck_curve_depth_pct']:.1f}~\\%")
        text = replace_macro(text, "PHDUCKMETWO",
                              f"{d['W2_ME']['duck_curve_depth_pct']:.1f}~\\%")
        shift_me = (d['W2_ME']['duck_curve_depth_pct']
                     - d['W1_ME']['duck_curve_depth_pct'])
        text = replace_macro(text, "PHDUCKMESHIFT", f"+{shift_me:.1f}~pp")

    # NEW v1.6: drift-weighted benchmark headlines (Step 3 outputs)
    if DW_BENCH.exists():
        b = json.loads(DW_BENCH.read_text())
        windows = b.get("windows", {})
        for label, suffix in (("W0", "WZ"), ("W1", "WO"), ("W2", "WT")):
            if label not in windows:
                continue
            w = windows[label]
            text = replace_macro(text, f"PHFROZEN{suffix}",
                                  f"{w['frozen']['overall_mape_pct']:.2f}~\\%")
            text = replace_macro(text, f"PHDW{suffix}",
                                  f"{w['drift_weighted']['overall_mape_pct']:.2f}~\\%")
            text = replace_macro(text, f"PHORACLE{suffix}",
                                  f"{w['oracle']['overall_mape_pct']:.2f}~\\%")
        # Gap-closed headline for W2.
        if "W2" in windows:
            w = windows["W2"]
            gap = w["frozen"]["overall_mape_pct"] - w["drift_weighted"]["overall_mape_pct"]
            oracle_gap = w["frozen"]["overall_mape_pct"] - w["oracle"]["overall_mape_pct"]
            if oracle_gap > 1e-6:
                pct_closed = 100.0 * gap / oracle_gap
                text = replace_macro(text, "PHDWGAPCLOSED",
                                      f"{pct_closed:.0f}~\\%")

    PREAMBLE.write_text(text)
    print(f"Wrote {PREAMBLE}")
    n_tbd = text.count(TBD)
    print(f"Remaining TBD placeholders: {n_tbd}")


if __name__ == "__main__":
    main()
