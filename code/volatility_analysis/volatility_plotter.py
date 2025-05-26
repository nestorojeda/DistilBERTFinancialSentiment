"""
Volatility Plotter Module

This module contains the VolatilityPlotter class for creating various types of financial 
market volatility plots and visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from typing import Optional
from scipy import stats
from matplotlib.patches import Rectangle


class VolatilityPlotter:
    """
    A class for creating various types of financial market volatility plots.
    
    This class encapsulates all plotting functionality for visualizing financial data,
    including volatility trends, sentiment analysis, and prediction results.
    """
    
    def __init__(self, style: str = "whitegrid", font_family: str = "sans-serif"):
        """
        Initialize the plotter with style preferences.
        
        Args:
            style: Seaborn style to use for plots.
            font_family: Font family for plot text.
        """
        self.style = style
        self.font_family = font_family
        self._setup_style()
    
    def _setup_style(self):
        """Set up the plotting style and font preferences."""
        sns.set_theme(style=self.style)
        plt.rcParams['font.family'] = self.font_family
        plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
    
    def get_sentiment_color(self, sentiment_value: float) -> str:
        """
        Get RGB color for a sentiment value using a gradient mapping.
        
        Args:
            sentiment_value: Sentiment score between -1 and 1.
        
        Returns:
            Hex color string representing the sentiment color.
        """
        # Normalize sentiment to [0, 1] range for color mapping
        # Map -1 to 0 (red), 0 to 0.5 (yellow), 1 to 1 (green)
        normalized_value = (sentiment_value + 1) / 2
        
        if normalized_value <= 0.5:  # -1 to 0 range: red to yellow
            red = 1.0
            green = normalized_value * 2  # 0 to 1
            blue = 0.0
        else:  # 0 to 1 range: yellow to green
            red = 2 * (1 - normalized_value)  # 1 to 0
            green = 1.0
            blue = 0.0

        rgb = (red, green, blue)    
        return '#' + ''.join(f'{int(c * 255):02x}' for c in rgb)
    
    def plot_volatility_news_count(self, merged_df: pd.DataFrame, 
                                  market_name: str = 'Market',
                                  save_path: Optional[str] = None,
                                  show_plot: bool = True) -> None:
        """
        Plot volatility and news article counts.
        
        Args:
            merged_df: DataFrame with merged volatility and news data.
            market_name: Name of the market for plot titles and labels.
            save_path: Path to save the plot image, or None to skip saving.
            show_plot: Whether to display the plot.
        """
        # Set the style with improved aesthetics
        self._setup_style()

        # Create figure and axes with better size
        fig, ax1 = plt.subplots(figsize=(16, 8), dpi=100)
        ax2 = ax1.twinx()  # Create a twin y-axis for volatility

        # Calculate better marker scaling to avoid extremely large markers
        max_count = merged_df['count'].max()
        min_size = 30
        max_size = 200
        size_scale = (max_size - min_size) / max_count if max_count > 0 else 1

        # Plot news article counts (left y-axis, as blue scatter)
        scatter = ax1.scatter(merged_df['date'], merged_df['count'], 
                            s=merged_df['count']*size_scale + min_size,  # Better sizing formula
                            color='#2176ae', 
                            alpha=0.7, 
                            edgecolor='white', 
                            linewidth=1.5, 
                            label='News Article Count', 
                            zorder=3)

        # Volatility line (right y-axis)
        sns.lineplot(x='date', y='Volatility_Smooth', 
                   data=merged_df, 
                   ax=ax2, 
                   color='#d7263d', 
                   label=f'{market_name} Volatility', 
                   linewidth=3, 
                   zorder=2)

        # Improved y-axis formatting
        ax1.set_ylabel("News Article Count", fontsize=16, color='#2176ae', fontweight='bold')
        ax1.tick_params(axis='y', labelcolor='#2176ae', labelsize=14)
        ax1.set_xlabel("Date", fontsize=16, fontweight='bold')
        ax1.set_ylim(0, merged_df['count'].max() * 1.2)  # Set reasonable y-axis limits

        # Format right y-axis (volatility)
        ax2.set_ylabel(f"{market_name} Volatility", fontsize=16, color='#d7263d', fontweight='bold')
        ax2.tick_params(axis='y', labelcolor='#d7263d', labelsize=14)
        ax2.yaxis.set_major_locator(MaxNLocator(nbins=8))

        # Add grid for better readability
        ax1.grid(axis='y', linestyle='--', alpha=0.3)

        # Annotate article counts more elegantly
        for i, row in merged_df.iterrows():
            if row['count'] > 0:
                # Only annotate counts above a threshold to avoid cluttering
                if row['count'] >= max(1, max_count * 0.1):  # Annotate counts that are at least 10% of max count
                    ax1.text(row['date'], row['count'] + max_count * 0.05,  # Position slightly above point
                            str(row['count']), 
                            color='#1b2a41', 
                            ha='center', 
                            va='bottom', 
                            fontsize=11, 
                            fontweight='bold', 
                            alpha=0.9)

        # Improved X-axis formatting with proper date range
        ax1.xaxis.set_major_locator(mdates.MonthLocator())
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))  # Include year in format
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha='right', fontsize=12)  # Rotate labels for better readability

        # Calculate the date range for dynamic title
        start_year = merged_df['date'].min().year
        end_year = merged_df['date'].max().year
        year_range = f"{start_year}" if start_year == end_year else f"{start_year}-{end_year}"

        # Title and legend with dynamic year range
        plt.title(f"{market_name} Volatility and News Article Counts ({year_range})", 
                 fontsize=20, fontweight='bold', pad=20)

        # Improved legend positioning and formatting
        ax1.legend(loc='upper left', fontsize=14, frameon=True, fancybox=True, borderpad=1)
        ax2.legend(loc='upper right', fontsize=14, frameon=True, fancybox=True, borderpad=1)

        # Add a subtle background color for better contrast
        fig.patch.set_facecolor('#f8f9fa')

        # Add tight layout and adjust to prevent any overlapping elements
        plt.tight_layout(pad=3)

        # Add a subtle grid background for the entire plot
        ax1.grid(True, linestyle='--', alpha=0.2)

        # Save the plot if a path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # Show the plot
        if show_plot:
            plt.show()
        else:
            plt.close()

    def plot_volatility_sentiment(self, merged_df: pd.DataFrame, 
                                 market_name: str = 'Market',
                                 save_path: Optional[str] = None,
                                 show_plot: bool = True) -> None:
        """
        Plot volatility and news sentiment.
        
        Args:
            merged_df: DataFrame with merged volatility and sentiment data.
            market_name: Name of the market for plot titles and labels.
            save_path: Path to save the plot image, or None to skip saving.
            show_plot: Whether to display the plot.
        """
        # Define sentiment colors and labels
        label_map = {"negative": -1, "neutral": 0, "positive": 1}
        inverse_label_map = {v: k for k, v in label_map.items()}

        # Create a figure with two y-axes
        fig, ax1 = plt.subplots(figsize=(14, 8))
        ax2 = ax1.twinx()  # Create a second y-axis

        # Plot volatility line
        ax2.plot(merged_df['date'], merged_df['Volatility_Smooth'], color='#2176ae', linewidth=2.5, 
                 label=f'{market_name} Volatility', alpha=0.8)

        # Plot sentiment markers on the volatility line
        for i, row in merged_df.iterrows():
            # Skip rows with NaN sentiment values
            if pd.isna(row['sentiment']):
                continue
            else:
                sentiment_value = row['sentiment']
                # Use the sentiment value to determine the color
                ax2.scatter(row['date'], row['Volatility_Smooth'], 
                            color=self.get_sentiment_color(sentiment_value), 
                            s=100, 
                            zorder=5,
                            edgecolor='white', 
                            linewidth=1.5)

        # Add colored marker samples for the legend
        for sentiment_value in [-1, 0, 1]:
            ax1.scatter([], [], color=self.get_sentiment_color(sentiment_value), 
                        label=f"{inverse_label_map[sentiment_value].capitalize()} Sentiment",
                        s=100, edgecolor='white', linewidth=1.5)

        # Format axes
        ax1.set_xlabel("Date", fontsize=15, fontweight='bold')
        ax2.set_ylabel("Volatility", fontsize=15, fontweight='bold', color='#2176ae')

        # Hide the y-axis on the left since we're not using it for data
        ax1.set_yticks([])
        ax1.spines['left'].set_visible(False)

        # Format date axis
        ax1.xaxis.set_major_locator(mdates.MonthLocator())
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))  # Include year in format
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha='right', fontsize=12)  # Rotate for better visibility

        # Add grid for better readability
        ax2.grid(axis='y', linestyle='--', alpha=0.3)

        # Calculate the date range for dynamic title
        start_year = merged_df['date'].min().year
        end_year = merged_df['date'].max().year
        year_range = f"{start_year}" if start_year == end_year else f"{start_year}-{end_year}"

        # Add title
        plt.title(f"{market_name} Volatility and News Sentiment ({year_range})", fontsize=20, fontweight='bold', pad=20)

        # Create combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=13, frameon=True, fancybox=True)

        # Add a subtle background color for better contrast
        fig.patch.set_facecolor('#f8f9fa')

        plt.tight_layout(pad=2)
        
        # Save the plot if a path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # Show the plot
        if show_plot:
            plt.show()
        else:
            plt.close()

    def plot_prediction_results(self, test_dates: np.ndarray, y_test_inv: np.ndarray, y_pred_inv: np.ndarray,
                               market_name: str = 'Market',
                               save_path: Optional[str] = None,
                               show_plot: bool = True) -> None:
        """
        Plot predicted vs actual volatility.
        
        Args:
            test_dates: Array of dates for the x-axis.
            y_test_inv: Array of actual volatility values.
            y_pred_inv: Array of predicted volatility values.
            market_name: Name of the market for plot titles and labels.
            save_path: Path to save the plot image, or None to skip saving.
            show_plot: Whether to display the plot.
        """
        plt.figure(figsize=(14, 6))
        plt.plot(test_dates, y_test_inv, label='Actual Volatility', color='#2176ae', linewidth=2)
        plt.plot(test_dates, y_pred_inv, label='Predicted Volatility (LSTM)', color='#d7263d', linewidth=2, linestyle='--')

        # Format date axis
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
        plt.xticks(rotation=45)

        plt.title(f'LSTM Volatility Prediction vs. Actual ({market_name})', fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=14)
        plt.ylabel('Volatility', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best', fontsize=12)

        # Add tight layout to prevent label cutoff
        plt.tight_layout()
        
        # Save the plot if a path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # Show the plot
        if show_plot:
            plt.show()
        else:
            plt.close()

    def plot_sentiment_distribution(self, merged_df: pd.DataFrame, market_name: str, 
                                   save_path: Optional[str] = None, show_plot: bool = True) -> None:
        """
        Plot sentiment distribution with gradient colors.
        
        Args:
            merged_df: DataFrame containing sentiment data.
            market_name: Name of the market for plot title.
            save_path: Path to save the plot image, or None to skip saving.
            show_plot: Whether to display the plot.
        """
        # Create a figure for sentiment distribution with custom colors
        plt.figure(figsize=(10, 6))
        
        # Create histogram data
        counts, bins, patches = plt.hist(merged_df['sentiment'].dropna(), bins=30, alpha=0.7, edgecolor='black')
        
        # Apply gradient colors based on bin centers
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        # Normalize bin centers to [0, 1] range for color mapping
        # Map -1 to 0 (red), 0 to 0.5 (yellow), 1 to 1 (green)
        normalized_centers = (bin_centers + 1) / 2
        
        # Apply colors to patches
        for i, (patch, center) in enumerate(zip(patches, normalized_centers)):
            if center <= 0.5:  # -1 to 0 range: red to yellow
                red = 1.0
                green = center * 2  # 0 to 1
                blue = 0.0
            else:  # 0 to 1 range: yellow to green
                red = 2 * (1 - center)  # 1 to 0
                green = 1.0
                blue = 0.0
                
            patch.set_facecolor((red, green, blue))
        
        # Add KDE overlay
        sentiment_data = merged_df['sentiment'].dropna()
        if len(sentiment_data) > 1:
            density = stats.gaussian_kde(sentiment_data)
            xs = np.linspace(sentiment_data.min(), sentiment_data.max(), 200)
            plt.plot(xs, density(xs) * len(sentiment_data) * (bins[1] - bins[0]), 
                color='black', linewidth=2, alpha=0.8, label='KDE')
        
        plt.title(f"Sentiment Distribution for {market_name}", fontsize=16, fontweight='bold')
        plt.xlabel("Sentiment Score", fontsize=14)
        plt.ylabel("Frequency", fontsize=14)
        plt.grid(True, alpha=0.3)
        
        # Add color legend
        legend_elements = [
            Rectangle((0, 0), 1, 1, facecolor='red', alpha=0.7, label='Negative (-1)'),
            Rectangle((0, 0), 1, 1, facecolor='yellow', alpha=0.7, label='Neutral (0)'),
            Rectangle((0, 0), 1, 1, facecolor='green', alpha=0.7, label='Positive (1)')
        ]
        plt.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        
        # Save the plot if a path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # Show the plot
        if show_plot:
            plt.show()
        else:
            plt.close()

    def plot_feature_correlation(self, df: pd.DataFrame, feature: str, target: str = 'Volatility_Smooth', market_name: str = 'Market', 
                                 save_path: Optional[str] = None, 
                                 show_plot: bool = True) -> None:
        
        """
        Plot feature correlation with the target variable.

        Args:
            df: DataFrame containing the features and target variable.
            feature: Name of the feature to plot against the target.
            target: Name of the target variable (default is 'Volatility_Smooth').
            market_name: Name of the market for plot title.
            save_path: Path to save the plot image, or None to skip saving.
            show_plot: Whether to display the plot.
        """
        # Set the style with improved aesthetics
        self._setup_style()
        # Create a figure
        plt.figure(figsize=(12, 6))
        # Scatter plot of the feature against the target variable
        sns.scatterplot(data=df, x=feature, y=target, color='#2176ae', alpha=0.7, edgecolor='white', linewidth=1.5) 
        # Fit a linear regression line
        sns.regplot(data=df, x=feature, y=target, scatter=False, color='#d7263d', line_kws={'linewidth': 2})
        # Set labels and title
        plt.xlabel(feature, fontsize=14, fontweight='bold')
        plt.ylabel(target, fontsize=14, fontweight='bold')
        plt.title(f"{market_name} {feature} vs {target}", fontsize=16, fontweight='bold')
        # Add grid for better readability
        plt.grid(True, linestyle='--', alpha=0.3)
        # Add tight layout to prevent label cutoff
        plt.tight_layout()
        # Save the plot if a path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        # Show the plot
        if show_plot:
            plt.show()
        else:
            plt.close()
