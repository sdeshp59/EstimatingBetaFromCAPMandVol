from data_processor import PreProcessor
from feature_eng import FeatureEngineer
from analysis import Analysis
from visualizations import Visualizations


def main():
    preprocessor = PreProcessor()
    sample, crsp = preprocessor.get_data()
    
    feature_eng = FeatureEngineer(sample, crsp)
    betas = feature_eng.compute_sampled_betas()
    betas = feature_eng.calculate_volatilities(betas, lookback_months=12)
    betas = feature_eng.calculate_volatilities(betas, lookback_months=24)
    betas = feature_eng.calculate_volatilities(betas, lookback_months=36)
    
    analyzer = Analysis(betas)
    desc_stats = analyzer.get_descriptive_stats_by_industry(output_path='outputs/descriptive_stats_by_industry.csv')
    print(desc_stats)
    
    annual_stats = analyzer.get_annual_stats()
    
    missing_betas = analyzer.analyze_missing_betas()
    missing_summary = missing_betas.groupby('beta_period')['missing_pct'].describe()
    print(missing_summary)

    vol_trends = analyzer.get_volatility_trends()

    visualizer = Visualizations(betas, annual_stats)
    visualizer.plot_all_beta_periods_mean(save_path='outputs/beta_mean_trends.png')
    visualizer.plot_all_beta_periods_std(save_path='outputs/beta_std_trends.png')
    visualizer.plot_volatility_trends(vol_trends, save_path='outputs/volatility_trends.png')
    visualizer.plot_missing_betas_heatmap(missing_betas, beta_period='beta_12m', save_path='outputs/missing_betas_12m.png')

    betas["mktcap"] = betas["PRC"].abs() * betas["SHROUT"]

    ew_beta, vw_beta = analyzer.form_quintile_portfolios(
        sort_col="beta_12m",
        ret_col="excess_stock",
        beta_col="beta_12m",
        mktcap_col="mktcap"
    )

    spread_ew_beta = analyzer.compute_spread(ew_beta, "ew_ret")
    spread_vw_beta = analyzer.compute_spread(vw_beta, "vw_ret")
    
    ew_ivol, vw_ivol = analyzer.form_quintile_portfolios(
        sort_col="IVOL_12m",
        ret_col="excess_stock",
        beta_col="beta_12m",
        mktcap_col="mktcap"
    )

    spread_ew_ivol = analyzer.compute_spread(ew_ivol, "ew_ret")
    spread_vw_ivol = analyzer.compute_spread(vw_ivol, "vw_ret")

    print(f"Equal-weighted IVOL spread (Q5-Q1): {spread_ew_ivol:.6f}")
    print(f"Value-weighted IVOL spread (Q5-Q1): {spread_vw_ivol:.6f}")


if __name__ == '__main__':
    main()