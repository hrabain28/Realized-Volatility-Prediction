import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime

def plot_target_distribution(targets_df):
    """Analyse la distribution des targets (volatilité)"""
    if targets_df is None or 'target' not in targets_df.columns:
        print("Pas de données targets disponibles")
        return
        
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Distribution des Targets (Volatilité Réalisée)', fontsize=16)
    
    # Distribution brute
    axes[0,0].hist(targets_df['target'], bins=50, alpha=0.7, edgecolor='black')
    axes[0,0].set_title('Distribution des targets')
    axes[0,0].set_xlabel('Volatilité réalisée')
    axes[0,0].set_ylabel('Fréquence')
    
    # Distribution log
    log_targets = np.log1p(targets_df['target'])
    axes[0,1].hist(log_targets, bins=50, alpha=0.7, edgecolor='black', color='orange')
    axes[0,1].set_title('Distribution log des targets')
    axes[0,1].set_xlabel('log(1 + Volatilité réalisée)')
    axes[0,1].set_ylabel('Fréquence')
    
    # Box plot par stock_id (top 10)
    top_stocks = targets_df['stock_id'].value_counts().head(10).index
    top_data = targets_df[targets_df['stock_id'].isin(top_stocks)]
    sns.boxplot(data=top_data, x='stock_id', y='target', ax=axes[1,0])
    axes[1,0].set_title('Distribution par stock (Top 10)')
    axes[1,0].tick_params(axis='x', rotation=45)
    
    # Statistiques descriptives
    stats_text = f"""Statistiques des targets:
    Count: {targets_df['target'].count():,}
    Mean: {targets_df['target'].mean():.4f}
    Std: {targets_df['target'].std():.4f}
    Min: {targets_df['target'].min():.4f}
    Max: {targets_df['target'].max():.4f}
    
    Skewness: {targets_df['target'].skew():.4f}
    Kurtosis: {targets_df['target'].kurtosis():.4f}
    
    Nombre de stocks: {targets_df['stock_id'].nunique()}
    Nombre de time_id: {targets_df['time_id'].nunique()}
    """
    
    axes[1,1].text(0.1, 0.5, stats_text, transform=axes[1,1].transAxes, 
                   fontsize=10, verticalalignment='center',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    axes[1,1].axis('off')
    
    plt.tight_layout()
    plt.show()

def plot_book_analysis(book_df):
    """Analyse les données du carnet d'ordres"""
    if book_df is None:
        print("Pas de données book disponibles")
        return
        
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Analyse du Carnet d\'Ordres (Order Book)', fontsize=16)
    
    # Spread bid-ask level 1
    book_df['spread_1'] = book_df['ask_price1'] - book_df['bid_price1']
    axes[0,0].hist(book_df['spread_1'].dropna(), bins=50, alpha=0.7, edgecolor='black')
    axes[0,0].set_title('Distribution du Spread (Level 1)')
    axes[0,0].set_xlabel('Ask Price 1 - Bid Price 1')
    axes[0,0].set_ylabel('Fréquence')
    
    # Evolution temporelle des prix (échantillon d'un stock)
    if 'stock_id' in book_df.columns and book_df['stock_id'].nunique() > 0:
        sample_stock = book_df['stock_id'].iloc[0]
        stock_data = book_df[book_df['stock_id'] == sample_stock].head(1000)
        
        axes[0,1].plot(stock_data['seconds_in_bucket'], stock_data['bid_price1'], 
                      label='Bid Price 1', alpha=0.7)
        axes[0,1].plot(stock_data['seconds_in_bucket'], stock_data['ask_price1'], 
                      label='Ask Price 1', alpha=0.7)
        axes[0,1].set_title(f'Evolution des prix - Stock {sample_stock}')
        axes[0,1].set_xlabel('Secondes dans le bucket')
        axes[0,1].set_ylabel('Prix normalisé')
        axes[0,1].legend()
    
    # Tailles des ordres
    axes[0,2].scatter(book_df['bid_size1'], book_df['ask_size1'], alpha=0.5, s=1)
    axes[0,2].set_title('Tailles des ordres Bid vs Ask (Level 1)')
    axes[0,2].set_xlabel('Bid Size 1')
    axes[0,2].set_ylabel('Ask Size 1')
    axes[0,2].set_xscale('log')
    axes[0,2].set_yscale('log')
    
    # Distribution des secondes dans le bucket
    axes[1,0].hist(book_df['seconds_in_bucket'], bins=50, alpha=0.7, edgecolor='black')
    axes[1,0].set_title('Distribution des secondes dans le bucket')
    axes[1,0].set_xlabel('Secondes')
    axes[1,0].set_ylabel('Fréquence')
    
    # Corrélation entre les niveaux de prix
    price_cols = ['bid_price1', 'bid_price2', 'ask_price1', 'ask_price2']
    available_price_cols = [col for col in price_cols if col in book_df.columns]
    if len(available_price_cols) > 1:
        corr_matrix = book_df[available_price_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1,1])
        axes[1,1].set_title('Corrélation entre niveaux de prix')
    
    # Stats par time_id (échantillon)
    if 'time_id' in book_df.columns:
        time_stats = book_df.groupby('time_id').agg({
            'spread_1': 'mean',
            'bid_size1': 'mean',
            'ask_size1': 'mean'
        }).head(20)
        
        axes[1,2].plot(time_stats.index, time_stats['spread_1'], marker='o', markersize=3)
        axes[1,2].set_title('Spread moyen par Time ID (échantillon)')
        axes[1,2].set_xlabel('Time ID')
        axes[1,2].set_ylabel('Spread moyen')
        axes[1,2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()

def plot_trade_analysis(trade_df):
    """Analyse les données de trades"""
    if trade_df is None:
        print("Pas de données trade disponibles")
        return
        
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Analyse des Trades Exécutés', fontsize=16)
    
    # Distribution des tailles de trades
    axes[0,0].hist(trade_df['size'], bins=50, alpha=0.7, edgecolor='black')
    axes[0,0].set_title('Distribution des tailles de trades')
    axes[0,0].set_xlabel('Taille (nombre d\'actions)')
    axes[0,0].set_ylabel('Fréquence')
    axes[0,0].set_yscale('log')
    
    # Distribution du nombre d'ordres
    axes[0,1].hist(trade_df['order_count'], bins=30, alpha=0.7, edgecolor='black', color='orange')
    axes[0,1].set_title('Distribution du nombre d\'ordres par trade')
    axes[0,1].set_xlabel('Nombre d\'ordres')
    axes[0,1].set_ylabel('Fréquence')
    
    # Relation taille vs prix
    axes[1,0].scatter(trade_df['size'], trade_df['price'], alpha=0.5, s=1)
    axes[1,0].set_title('Relation Taille vs Prix')
    axes[1,0].set_xlabel('Taille')
    axes[1,0].set_ylabel('Prix')
    axes[1,0].set_xscale('log')
    
    # Evolution temporelle des trades (échantillon)
    if 'stock_id' in trade_df.columns and trade_df['stock_id'].nunique() > 0:
        sample_stock = trade_df['stock_id'].iloc[0]
        stock_trades = trade_df[trade_df['stock_id'] == sample_stock].head(500)
        
        axes[1,1].scatter(stock_trades['seconds_in_bucket'], stock_trades['price'], 
                         s=stock_trades['size']/10, alpha=0.6)
        axes[1,1].set_title(f'Trades dans le temps - Stock {sample_stock}')
        axes[1,1].set_xlabel('Secondes dans le bucket')
        axes[1,1].set_ylabel('Prix')
    
    plt.tight_layout()
    plt.show()

def generate_summary_report(loader, book_data, trade_data, target_data):
    """Génère un rapport de synthèse"""
    print("\n" + "="*60)
    print("RAPPORT DE SYNTHÈSE - DONNÉES OPTIVER")
    print("="*60)
    
    print(f"\n📊 APERÇU GÉNÉRAL:")
    print(f"   • Données book disponibles: {'✓' if book_data is not None else '✗'}")
    print(f"   • Données trade disponibles: {'✓' if trade_data is not None else '✗'}")
    print(f"   • Données target disponibles: {'✓' if target_data is not None else '✗'}")
    
    if target_data is not None:
        print(f"\n🎯 TARGETS (VOLATILITÉ):")
        print(f"   • Nombre d'observations: {len(target_data):,}")
        print(f"   • Nombre de stocks: {target_data['stock_id'].nunique()}")
        print(f"   • Nombre de time buckets: {target_data['time_id'].nunique()}")
        print(f"   • Volatilité moyenne: {target_data['target'].mean():.4f}")
        print(f"   • Volatilité médiane: {target_data['target'].median():.4f}")
        print(f"   • Écart-type: {target_data['target'].std():.4f}")
        
    if book_data is not None:
        print(f"\n📖 CARNET D'ORDRES:")
        print(f"   • Nombre d'observations: {len(book_data):,}")
        print(f"   • Nombre de stocks: {book_data['stock_id'].nunique()}")
        print(f"   • Fréquence temporelle: {book_data['seconds_in_bucket'].max()} secondes max")
        spread_mean = (book_data['ask_price1'] - book_data['bid_price1']).mean()
        print(f"   • Spread moyen (Level 1): {spread_mean:.6f}")
        
    if trade_data is not None:
        print(f"\n💹 TRADES EXÉCUTÉS:")
        print(f"   • Nombre de trades: {len(trade_data):,}")
        print(f"   • Nombre de stocks: {trade_data['stock_id'].nunique()}")
        print(f"   • Taille moyenne des trades: {trade_data['size'].mean():.0f}")
        print(f"   • Volume total: {trade_data['size'].sum():,}")
        print(f"   • Ordres moyens par trade: {trade_data['order_count'].mean():.1f}")
    
    print(f"\n🔍 PROCHAINES ÉTAPES RECOMMANDÉES:")
    print(f"   1. Feature Engineering sur les données microstructure")
    print(f"   2. Analyse de la corrélation book-trade-volatilité") 
    print(f"   3. Création d'indicateurs techniques haute fréquence")
    print(f"   4. Modélisation de la volatilité à 10 minutes")
    print(f"   5. Validation temporelle des modèles")
    
    print("="*60)

# Fonction principale pour les visualisations
def visualize_data(loader, book_data, trade_data, target_data):
    """Lance toutes les visualisations"""
    print("=== GÉNÉRATION DES VISUALISATIONS ===\n")
    
    if target_data is not None:
        print("1. Analyse des targets...")
        plot_target_distribution(target_data)
    
    if book_data is not None:
        print("2. Analyse du carnet d'ordres...")
        plot_book_analysis(book_data)
    
    if trade_data is not None:
        print("3. Analyse des trades...")
        plot_trade_analysis(trade_data)
    
    print("4. Génération du rapport de synthèse...")
    generate_summary_report(loader, book_data, trade_data, target_data)
    
    print("\n=== VISUALISATIONS TERMINÉES ===")

# Utilisation après avoir chargé les données:
# visualize_data(loader, book_data, trade_data, target_data)