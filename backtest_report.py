from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


class BacktestReporter:
    """Write console, CSV, chart, and Markdown backtest outputs."""

    def __init__(self, engine: Any):
        """Store engine context for output paths and config."""
        self.engine = engine
        self.output_dir = engine.output_dir

    def report(self, results: Any) -> None:
        """Print console report and write all output files."""
        self._write_trades_csv(results)
        self._plot_equity(results)
        self._plot_monthly(results)
        self._plot_hourly(results)
        self._plot_drawdown(results)
        self._write_markdown(results)
        self._print_console(results)
        print("\nAll output files saved to: ./backtest_output/")
        for name in ["backtest_results.csv", "backtest_summary.md", "backtest_equity_curves.png", "backtest_monthly_heatmap.png", "backtest_win_rate_by_hour.png", "backtest_drawdown.png"]:
            print(f"  - {name}")

    def _write_trades_csv(self, results: Any) -> None:
        """Write one row per completed trade."""
        rows = [t for r in results.strategies.values() for t in r.trades]
        cols = ["date", "time", "strategy", "symbol", "side", "entry_price", "exit_price", "qty", "gross_pnl", "net_pnl", "outcome", "holding_minutes", "signal_score", "rejection_reason"]
        pd.DataFrame(rows, columns=cols).to_csv(self.output_dir / "backtest_results.csv", index=False)

    def _plot_equity(self, results: Any) -> None:
        """Save cumulative P&L chart."""
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 6))
        for name, res in results.strategies.items():
            res.daily_pnl.cumsum().plot(label=name)
        plt.axhline(0, color="black", linewidth=0.8)
        plt.title("Backtest Equity Curves")
        plt.ylabel("Cumulative Net P&L (INR)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.output_dir / "backtest_equity_curves.png", dpi=150)
        plt.close()

    def _plot_monthly(self, results: Any) -> None:
        """Save monthly P&L heatmaps."""
        import matplotlib.pyplot as plt
        n = len(results.strategies)
        if n == 0: return
        fig, axes = plt.subplots(1, n, figsize=(min(16, 4 * n), 6), squeeze=False)
        for ax, (name, res) in zip(axes[0], results.strategies.items()):
            vals = res.monthly["Net P&L"].to_numpy().reshape(-1, 1) if not res.monthly.empty else np.array([[0]])
            vmax = max(abs(vals).max(), 1)
            ax.imshow(vals, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
            ax.set_title(name)
            labels = res.monthly["Month"].tolist() if not res.monthly.empty else ["No trades"]
            ax.set_yticks(range(len(labels)), labels=labels)
            ax.set_xticks([0], ["P&L"])
            for y, v in enumerate(vals[:, 0]):
                ax.text(0, y, f"{v:,.0f}", ha="center", va="center", fontsize=8)
        plt.tight_layout()
        plt.savefig(self.output_dir / "backtest_monthly_heatmap.png", dpi=150)
        plt.close()

    def _plot_hourly(self, results: Any) -> None:
        """Save win-rate-by-hour chart."""
        import matplotlib.pyplot as plt
        n = len(results.strategies)
        if n == 0: return
        rows = (n + 1) // 2
        fig, axes = plt.subplots(rows, 2, figsize=(12, 4 * rows), squeeze=False)
        hours = list(range(9, 16))
        for ax, (name, res) in zip(axes.ravel(), results.strategies.items()):
            wr = res.metrics["win_rate_by_hour"]
            ax.bar(hours, [wr.get(h, 0) for h in hours], color="#4c78a8")
            ax.set_ylim(0, 100)
            ax.set_title(name)
            ax.set_ylabel("Win Rate %")
        plt.tight_layout()
        plt.savefig(self.output_dir / "backtest_win_rate_by_hour.png", dpi=150)
        plt.close()

    def _plot_drawdown(self, results: Any) -> None:
        """Save drawdown curves."""
        import matplotlib.pyplot as plt
        n = len(results.strategies)
        if n == 0: return
        rows = (n + 1) // 2
        fig, axes = plt.subplots(rows, 2, figsize=(12, 4 * rows), squeeze=False)
        for ax, (name, res) in zip(axes.ravel(), results.strategies.items()):
            equity = res.daily_pnl.cumsum()
            dd = equity - equity.cummax()
            ax.fill_between(dd.index, dd.to_numpy(), 0, color="red", alpha=0.3)
            ax.plot(dd.index, dd.to_numpy(), color="red", linewidth=1)
            ax.set_title(name)
            ax.set_ylabel("Drawdown INR")
        plt.tight_layout()
        plt.savefig(self.output_dir / "backtest_drawdown.png", dpi=150)
        plt.close()

    def _write_markdown(self, results: Any) -> None:
        """Write quant-readable Markdown summary."""
        lines = ["# Backtest Summary", "", "## Executive Summary"]
        ranked = sorted(results.strategies.values(), key=lambda r: r.metrics["sharpe"], reverse=True)
        lines += [
            f"- Best Sharpe: **{ranked[0].name}** at **{ranked[0].metrics['sharpe']:.2f}**.",
            f"- Best net P&L: **{max(results.strategies.values(), key=lambda r: r.metrics['total_net_pnl']).name}**.",
            f"- Most active strategy: **{max(results.strategies.values(), key=lambda r: r.metrics['total_trades']).name}**.",
            "- Synthetic option premiums are a proxy, not a substitute for real Sensex option chain data.",
            "- **Warning: Survivorship Bias.** This backtest only includes currently liquid symbols. Historical delistings or mergers are not modeled.",
            "- Strategies marked NOT READY should remain paper-only until validated on broker-grade intraday data.",
            "",
        ]
        for name, res in results.strategies.items():
            m = res.metrics
            lines += [f"## {name}", "", "| Metric | Value |", "|---|---|"]
            for k in ["total_net_pnl", "total_net_pnl_pct", "annualised_return_pct", "max_drawdown_inr", "max_drawdown_pct", "sharpe", "profit_factor", "win_rate_pct", "total_trades", "avg_r_multiple", "daily_loss_limit_days"]:
                lines.append(f"| {k.replace('_', ' ').title()} | {m[k]:.2f} |")
            lines += ["", "### Monthly Breakdown", self._markdown_table(res.monthly) if not res.monthly.empty else "No trades.", "", "### Top 3 Best Trades"]
            df = pd.DataFrame(res.trades)
            lines.append(self._markdown_table(df.sort_values("net_pnl", ascending=False).head(3)[["date", "symbol", "net_pnl", "outcome"]]) if not df.empty else "No trades.")
            lines += ["", "### Top 3 Worst Trades"]
            lines.append(self._markdown_table(df.sort_values("net_pnl").head(3)[["date", "symbol", "net_pnl", "outcome"]]) if not df.empty else "No trades.")
            lines += ["", "### Parameter Sensitivity", self._markdown_table(res.sensitivity) if res.sensitivity is not None else "Not run.", "", "### Quant Review Flags"]
            for k, v in (res.flags or {}).items():
                lines.append(f"- **{k}:** {v}")
            lines += ["", "### Honest Assessment", self._assessment(res), "", "### Recommended Next Steps", "- Validate against broker-grade 5m/15m historical candles.", "- Replace simulated AI scores with walk-forward trained models.", "- Re-run with realistic liquidity, taxes, and symbol-specific slippage.", "- Keep live capital disabled until out-of-sample performance is stable.", ""]
        (self.output_dir / "backtest_summary.md").write_text("\n".join(lines), encoding="utf-8")

    def _markdown_table(self, df: pd.DataFrame) -> str:
        """Render a Markdown table without optional tabulate dependency."""
        if df.empty:
            return "No rows."
        safe = df.copy()
        for col in safe.columns:
            if pd.api.types.is_float_dtype(safe[col]):
                safe[col] = safe[col].map(lambda x: f"{x:.2f}" if np.isfinite(x) else "inf")
        headers = [str(c) for c in safe.columns]
        lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
        for _, row in safe.iterrows():
            lines.append("| " + " | ".join(str(row[c]) for c in safe.columns) + " |")
        return "\n".join(lines)

    def _print_console(self, results: Any) -> None:
        """Print formatted console report."""
        print("\n================ BACKTEST REPORT ================")
        for name, res in results.strategies.items():
            m = res.metrics
            print(f"\n{name.upper()}")
            print(f"  Net P&L: INR {m['total_net_pnl']:,.2f} ({m['total_net_pnl_pct']:.2f}%) | Sharpe: {m['sharpe']:.2f} | PF: {m['profit_factor']:.2f}")
            print(f"  Trades: {m['total_trades']} / Signals: {m['total_signals']} | WR: {m['win_rate_pct']:.1f}% | Max DD: {m['max_drawdown_pct']:.1f}%")
            print(f"  Avg hold: {m['avg_holding_minutes']:.1f} min | Avg R: {m['avg_r_multiple']:.2f} | Verdict: {res.flags.get('Overall verdict') if res.flags else 'NA'}")
            print("  Sensitivity:")
            print(res.sensitivity.to_string(index=False) if res.sensitivity is not None else "  Not run")
        comp = []
        for name, res in results.strategies.items():
            m = res.metrics
            comp.append({"Strategy": name, "Trades": m["total_trades"], "Win Rate": f"{m['win_rate_pct']:.1f}%", "Net P&L": f"{m['total_net_pnl']:,.0f}", "Sharpe": f"{m['sharpe']:.2f}", "Max DD %": f"{m['max_drawdown_pct']:.1f}", "Verdict": res.flags.get("Overall verdict", "NA") if res.flags else "NA"})
        print("\nCOMPARISON TABLE")
        print(pd.DataFrame(comp).to_string(index=False))

    def _assessment(self, res: Any) -> str:
        """Return plain-English viability assessment."""
        v = (res.flags or {}).get("Overall verdict", "NOT READY")
        if v == "VIABLE":
            return "The strategy passes the mechanical review flags in this simulation, but still needs real intraday validation before live capital."
        if v == "NEEDS TUNING":
            return "The strategy has useful signal activity but one or two risk/quality checks need tuning before it should be trusted."
        return "The strategy is not ready. Either trade count, risk-adjusted return, drawdown, stability, or parameter robustness is insufficient."
