import matplotlib.pyplot as plt
import numpy as np


class Visualizations:
    def __init__(self, betas_df, annual_stats_df):
        """Initialize Visualizations class."""
        self.betas = betas_df
        self.annual_stats = annual_stats_df

    def plot_beta_mean_by_industry(self, beta_period='beta_12m', figsize=(14, 8), save_path=None):
        """Plot mean beta trends by industry over time (separate from std)."""
        plt.figure(figsize=figsize)

        for industry in self.annual_stats['industry'].unique():
            df = self.annual_stats[self.annual_stats['industry'] == industry]
            plt.plot(df['year'], df[f'{beta_period}_mean'],
                    label=industry, marker='o', markersize=4, linewidth=2)

        plt.title(f'Mean {beta_period} by Industry Over Time', fontsize=14, fontweight='bold')
        plt.xlabel('Year', fontsize=12)
        plt.ylabel('Mean Beta', fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def plot_beta_std_by_industry(self, beta_period='beta_12m', figsize=(14, 8), save_path=None):
        """Plot standard deviation of beta by industry over time (separate from mean)."""
        plt.figure(figsize=figsize)

        for industry in self.annual_stats['industry'].unique():
            df = self.annual_stats[self.annual_stats['industry'] == industry]
            plt.plot(df['year'], df[f'{beta_period}_std'],
                    label=industry, marker='s', markersize=4, linewidth=2)

        plt.title(f'Standard Deviation of {beta_period} by Industry Over Time', fontsize=14, fontweight='bold')
        plt.xlabel('Year', fontsize=12)
        plt.ylabel('Beta Standard Deviation', fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        plt.show()

    def plot_all_beta_periods_mean(self, figsize=(14, 18), save_path=None):
        """Plot mean beta for all three periods (12m, 24m, 36m) in subplots."""
        fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
        beta_periods = ['beta_12m', 'beta_24m', 'beta_36m']
        titles = ['12-Month Beta', '24-Month Beta', '36-Month Beta']

        for ax, beta_period, title in zip(axes, beta_periods, titles):
            for industry in self.annual_stats['industry'].unique():
                df = self.annual_stats[self.annual_stats['industry'] == industry]
                ax.plot(df['year'], df[f'{beta_period}_mean'],
                        label=industry, marker='o', markersize=3, linewidth=1.5)

            ax.set_title(f'Mean {title} by Industry', fontsize=12, fontweight='bold')
            ax.set_ylabel('Mean Beta', fontsize=11)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel('Year', fontsize=12)
        plt.suptitle('Beta Trends by Industry (Mean)', fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        plt.show()

    def plot_all_beta_periods_std(self, figsize=(14, 18), save_path=None):
        """Plot standard deviation of beta for all three periods (12m, 24m, 36m) in subplots."""
        fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
        beta_periods = ['beta_12m', 'beta_24m', 'beta_36m']
        titles = ['12-Month Beta', '24-Month Beta', '36-Month Beta']

        for ax, beta_period, title in zip(axes, beta_periods, titles):
            for industry in self.annual_stats['industry'].unique():
                df = self.annual_stats[self.annual_stats['industry'] == industry]
                ax.plot(df['year'], df[f'{beta_period}_std'],
                        label=industry, marker='s', markersize=3, linewidth=1.5)

            ax.set_title(f'Std Dev of {title} by Industry', fontsize=12, fontweight='bold')
            ax.set_ylabel('Beta Std Dev', fontsize=11)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel('Year', fontsize=12)
        plt.suptitle('Beta Dispersion by Industry (Standard Deviation)', fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def plot_volatility_trends(self, vol_trends_df, figsize=(14, 16), save_path=None):
        """Plot volatility component trends for all three lookback periods."""
        fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
        lookbacks = ["12m", "24m", "36m"]

        for ax, lb in zip(axes, lookbacks):
            ax.plot(vol_trends_df["year"], vol_trends_df[f"TVOL_{lb}"],
                    label="Total Volatility (TVOL)", linewidth=2, marker='o')
            ax.plot(vol_trends_df["year"], vol_trends_df[f"SVOL_{lb}"],
                    label="Systematic Volatility (SVOL)", linewidth=2, marker='s')
            ax.plot(vol_trends_df["year"], vol_trends_df[f"IVOL_{lb}"],
                    label="Idiosyncratic Volatility (IVOL)", linewidth=2, marker='^')

            ax.set_title(f"Volatility Components ({lb} lookback)", fontsize=12, fontweight='bold')
            ax.set_ylabel("Average Volatility", fontsize=11)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel("Year", fontsize=12)
        plt.suptitle("Trends in Stock Volatility Components", fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def plot_missing_betas_heatmap(self, missing_df, beta_period='beta_12m', figsize=(14, 10), save_path=None):
        """Plot heatmap of missing beta percentages by year and industry."""
        subset = missing_df[missing_df['beta_period'] == beta_period].copy()
        pivot = subset.pivot(index='industry', columns='year', values='missing_pct')

        plt.figure(figsize=figsize)
        im = plt.imshow(pivot.values, cmap='YlOrRd', aspect='auto', interpolation='nearest')

        plt.colorbar(im, label='Missing %')
        plt.yticks(range(len(pivot.index)), pivot.index, fontsize=9)
        plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=45, ha='right', fontsize=8)
        plt.xlabel('Year', fontsize=12)
        plt.ylabel('Industry', fontsize=12)
        plt.title(f'Missing {beta_period} by Year and Industry (%)', fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()
