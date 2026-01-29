import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_average_spend(sales_raw, output_path='../outputs/figures/average_spend.png'):
    """
    Gera e guarda o gráfico de média de gastos por tipo de cliente (Member vs Normal).
    """
    plt.figure(figsize=(10, 6))
    # No seu dataset a coluna de valor total chama-se 'Total' ou 'Sales'
    target_col = 'Total' if 'Total' in sales_raw.columns else 'Sales'
    
    avg_spend = sales_raw.groupby('Customer type')[target_col].mean()
    avg_spend.plot(kind='bar', color=['#3498db', '#e74c3c'])
    
    plt.title('Average Spend: Member vs Normal')
    plt.ylabel('Average Total')
    plt.xticks(rotation=0)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.show()
    print(f"📊 Gráfico guardado em: {output_path}")

def plot_product_performance(sales_raw, output_path='../outputs/figures/product_performance.png'):
    """
    Analisa a distribuição de vendas por linha de produto.
    """
    plt.figure(figsize=(12, 6))
    sns.countplot(y='Product line', data=sales_raw, order=sales_raw['Product line'].value_counts().index)
    plt.title('Performance por Linha de Produto')
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.show()
