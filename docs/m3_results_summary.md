# M3 Results Summary

This summary is based on the official M3 protocol run generated locally on 2026-03-18 with:

```bash
python -m src.engine.comparison_runner --config config/evaluation/m3_protocol.yaml
```

The evidence run id was `m3-comparison-20260318T143027Z` under `outputs/backtests/comparisons/`.

## Evidence Used

This write-up uses the following saved artifacts from that comparison run:

- `comparison_summary.json`
- `comparison_metrics.csv`
- `ranking_summary.json`
- `strategy_ranking.csv`
- `manifest.json`

These outputs are ignored from git, so this document records the milestone-level conclusion from the saved local artifacts rather than assuming committed result files exist in the repository.

## What Was Compared

The M3 run compared:

- `momentum_baseline`
- `momentum_faster_momentum`
- `momentum_slower_momentum`
- `buy_and_hold`
- `equal_weight`

The baselines came from the same official protocol as the momentum variants, so the comparison used one shared universe, one benchmark symbol, one date window, and one metric/export pipeline.

## What The Results Showed

The ranking output selected `equal_weight` as the preferred configuration.

Rounded from `comparison_metrics.csv` and `strategy_ranking.csv`:

| Rank | Run | strategy_type | cumulative_return | max_drawdown | sharpe_ratio |
| --- | --- | --- | --- | --- | --- |
| 1 | `equal_weight` | `baseline` | 3.835 | -0.496 | 0.0706 |
| 2 | `momentum_faster_momentum` | `momentum` | 1.870 | -0.340 | 0.0578 |
| 3 | `momentum_slower_momentum` | `momentum` | 1.564 | -0.258 | 0.0559 |
| 4 | `momentum_baseline` | `momentum` | 1.609 | -0.329 | 0.0542 |
| 5 | `buy_and_hold` | `baseline` | 1.439 | -0.351 | 0.0520 |

Supported comparisons:

- The current main strategy, `momentum_baseline`, beat `buy_and_hold` on `cumulative_return`, `max_drawdown`, and `sharpe_ratio`.
- `momentum_baseline` did not beat `equal_weight`. The equal-weight baseline had much higher `cumulative_return` and a higher `sharpe_ratio`, although it also had the deepest `max_drawdown`.
- The parameter variants materially changed the active strategy ranking. `momentum_faster_momentum` was the strongest momentum variant by both `cumulative_return` and `sharpe_ratio`.
- `momentum_slower_momentum` did not produce the highest active return, but it had the shallowest `max_drawdown` and lowest `volatility` among the momentum variants.
- The ranking layer therefore chose a passive baseline, not a momentum variant, as the preferred M3 configuration.

## Strengths

- The momentum family was still competitive. All three momentum runs finished ahead of `buy_and_hold` on both `cumulative_return` and `sharpe_ratio`.
- The variant results were interpretable rather than random-looking. The faster variant pushed returns higher, while the slower variant improved drawdown control.
- The comparison pipeline produced a complete evidence bundle: grouped run artifacts, comparison metrics, aligned equity exports, and a deterministic ranking summary.

## Weaknesses

- The current default momentum configuration was not the best active configuration and was not the overall winner.
- `equal_weight` outperformed every momentum variant on the ranking rule used by the pipeline, which means the active stock-selection edge is not yet strong enough to clear the passive basket baseline.
- `win_rate` was unavailable for the momentum runs in `comparison_metrics.csv`, so trade-level quality could not be compared as cleanly as return and drawdown behavior.

## Limitations

- This summary is based on one official protocol only: one universe, one benchmark, one transaction-cost model, and one date window from `config/evaluation/m3_protocol.yaml`.
- The ranking rule is intentionally simple. It is useful for repeatable selection, but it should not be treated as the only lens for decision-making.
- The saved outputs support comparison-level conclusions, but they do not by themselves explain why a specific variant won or lost on individual symbols or periods.
- Because the evidence artifacts are local outputs under `outputs/backtests/comparisons/`, another contributor will need to rerun the official comparison if those local files are not present in their workspace.

## Next-Step Implications

- Treat `equal_weight` as the current M3 preferred configuration because that is what the saved `ranking_summary.json` selected.
- Treat `momentum_faster_momentum` as the strongest active variant to study further, since it was the best-performing momentum run in this comparison.
- Use `equal_weight` as the practical hurdle for the next milestone. Beating `buy_and_hold` is no longer enough on its own.
- In the next iteration, focus on improving active edge without giving back too much drawdown control, because the current results show a clear return-versus-risk trade-off inside the momentum family.
