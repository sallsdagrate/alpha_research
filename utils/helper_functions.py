from itertools import product
import numpy as np
import pandas as pd
import statsmodels.api as sm

def make_walkforward_splits(index, train_years=3, test_months=6, step_months=6):
    """
    Expanding window splits based on calendar dates.
    """
    index = pd.DatetimeIndex(index).sort_values()
    start = index.min()
    end = index.max()

    splits = []
    train_end = start + pd.DateOffset(years=train_years)

    while True:
        test_end = train_end + pd.DateOffset(months=test_months)
        if test_end > end:
            break

        train_idx = index[index < train_end]
        test_idx  = index[(index >= train_end) & (index < test_end)]

        if len(train_idx) > 0 and len(test_idx) > 0:
            splits.append((train_idx, test_idx))

        train_end = train_end + pd.DateOffset(months=step_months)

    return splits


def evaluate_fold(train_sa, train_sb, test_sa, test_sb, window, entry, exit_):
    """
    Fit beta on train only, trade on test only.
    Rolling stats on test are seeded with tail of train spread to avoid warmup waste.
    """
    if exit_ >= entry:
        return None

    # fit hedge ratio on train only
    x_with_const = sm.add_constant(train_sb)
    hedge_model = sm.OLS(train_sa, x_with_const).fit()
    intercept, beta = hedge_model.params.iloc[0], hedge_model.params.iloc[1]

    # construct spreads
    spread_train = train_sa - beta * train_sb
    spread_test  = test_sa  - beta * test_sb

    # seed rolling window with tail of training spread
    seeded_spread = pd.concat([spread_train.iloc[-window:], spread_test])

    roll_mean = seeded_spread.rolling(window).mean().loc[spread_test.index]
    roll_std  = seeded_spread.rolling(window).std().loc[spread_test.index]
    zscore    = (spread_test - roll_mean) / roll_std

    position = pd.Series(np.nan, index=spread_test.index)
    position.loc[zscore > entry] = -1
    position.loc[zscore < -entry] = 1
    position.loc[zscore.abs() < exit_] = 0
    position = position.ffill().fillna(0)

    ret_a = np.log(test_sa / test_sa.shift(1))
    ret_b = np.log(test_sb / test_sb.shift(1))

    strategy = position.shift(1) * (ret_a - beta * ret_b)
    strategy = strategy.replace([np.inf, -np.inf], np.nan).dropna()

    if len(strategy) < 10 or strategy.std() == 0:
        return None

    cumret = np.exp(strategy.cumsum())
    total_return = cumret.iloc[-1] - 1
    annual_ret = np.exp(strategy.mean() * 252) - 1
    sharpe = (strategy.mean() / strategy.std()) * np.sqrt(252)
    max_dd = (cumret / cumret.cummax() - 1).min()

    return {
        'beta': beta,
        'total_return': total_return,
        'annual_ret': annual_ret,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'n_obs': len(strategy),
    }


def search_bollinger_params(
    sa, sb,
    windows=(20, 30, 50, 75, 100),
    entries=(1.5, 2.0, 2.5, 3.0),
    exits=(0.25, 0.5, 0.75, 1.0),
    train_years=3,
    test_months=6,
    step_months=6,
):
    splits = make_walkforward_splits(
        sa.index,
        train_years=train_years,
        test_months=test_months,
        step_months=step_months
    )

    results = []
    fold_results = []

    for window, entry, exit_ in product(windows, entries, exits):
        if exit_ >= entry:
            continue

        fold_rows = []

        for fold_num, (train_idx, test_idx) in enumerate(splits, start=1):
            train_sa, train_sb = sa.loc[train_idx], sb.loc[train_idx]
            test_sa, test_sb   = sa.loc[test_idx], sb.loc[test_idx]

            if len(train_sa) < window + 20 or len(test_sa) < window + 10:
                continue

            out = evaluate_fold(
                train_sa, train_sb,
                test_sa, test_sb,
                window, entry, exit_
            )

            if out is None:
                continue

            row = {
                'fold': fold_num,
                'window': window,
                'entry': entry,
                'exit': exit_,
                **out
            }
            fold_rows.append(row)
            fold_results.append(row)

        if len(fold_rows) == 0:
            continue

        fold_df = pd.DataFrame(fold_rows)

        results.append({
            'window': window,
            'entry': entry,
            'exit': exit_,
            'mean_total_return': fold_df['total_return'].mean(),
            'mean_annual_ret': fold_df['annual_ret'].mean(),
            'mean_sharpe': fold_df['sharpe'].mean(),
            'median_sharpe': fold_df['sharpe'].median(),
            'std_sharpe': fold_df['sharpe'].std(),
            'mean_max_dd': fold_df['max_dd'].mean(),
            'min_sharpe': fold_df['sharpe'].min(),
            'n_folds': len(fold_df),
        })

    results_df = pd.DataFrame(results).sort_values(
        ['mean_sharpe', 'mean_total_return'],
        ascending=[False, False]
    ).reset_index(drop=True)

    fold_results_df = pd.DataFrame(fold_results)
    return results_df, fold_results_df, splits

