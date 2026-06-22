from statsmodels.tsa.stattools import coint, adfuller

def pair_adf_test(sa, sb, name_a='A', name_b='B', significance=0.05):
    """
    Full cointegration analysis for two price series.
    Runs ADF on each series, then tests for cointegration.
    """
    sep = "=" * 55

    # --- ADF Tests ---
    def _adf(series, name):
        r = adfuller(series.dropna())
        stationary = r[1] < significance
        print(f"  {name}")
        print(f"    ADF Statistic : {r[0]:>10.4f}")
        print(f"    p-value       : {r[1]:>10.4f}  {'✓ stationary' if stationary else '✗ non-stationary'}")
        print(f"    Critical Values: 1%={r[4]['1%']:.3f}  5%={r[4]['5%']:.3f}  10%={r[4]['10%']:.3f}")
        return stationary

    print(sep)
    print(f"  ADF STATIONARITY TESTS")
    print(sep)
    a_stationary = _adf(sa, name_a)
    print()
    b_stationary = _adf(sb, name_b)

    # --- Cointegration ---
    print()
    print(sep)
    print(f"  COINTEGRATION TEST  ({name_a} ~ {name_b})")
    print(sep)

    if a_stationary or b_stationary:
        print(f"  ⚠  Warning: cointegration assumes both series are")
        print(f"     non-stationary. Results may be unreliable.")
        print()

    score, pvalue, crits = coint(sa.dropna(), sb.dropna())
    cointegrated = pvalue < significance

    print(f"  ADF Statistic  : {score:>10.4f}")
    print(f"  p-value        : {pvalue:>10.4f}  ({'✓ cointegrated' if cointegrated else '✗ not cointegrated'} at {significance*100:.0f}% level)")
    print(f"  Critical Values: 1%={crits[0]:.3f}  5%={crits[1]:.3f}  10%={crits[2]:.3f}")
    print(sep)

    return {"cointegrated": cointegrated, "pvalue": pvalue, "score": score}