import pandas as pd
import numpy as np
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import STL
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats
import os
import json
import warnings

warnings.filterwarnings('ignore')
plt.style.use('ggplot')

# --- Funções de Análise (da etapa anterior) ---
def tratar_outliers(serie, metodo='iqr', limite=1.5):
    serie_limpa = serie.copy()
    if metodo == 'iqr':
        Q1, Q3 = serie.quantile(0.25), serie.quantile(0.75)
        IQR = Q3 - Q1
        outliers_indices = np.where((serie < (Q1 - limite * IQR)) | (serie > (Q3 + limite * IQR)))[0]
    else:
        z_scores = np.abs(stats.zscore(serie))
        outliers_indices = np.where(z_scores > limite)[0]
    janela = 5
    for idx in outliers_indices:
        inicio, fim = max(0, idx - janela//2), min(len(serie), idx + janela//2 + 1)
        vizinhanca = [val for i, val in enumerate(serie.iloc[inicio:fim]) if i + inicio != idx]
        if vizinhanca: serie_limpa.iloc[idx] = np.median(vizinhanca)
    return serie_limpa

def visualizar_outliers(df_original, df_tratado, region_name):
    output_dir = "static/images"
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    filepath = f"{output_dir}/outliers_{region_name.lower()}.png"
    plt.figure(figsize=(14, 6))
    plt.plot(df_original.index, df_original, 'o-', alpha=0.7, label='Série Original', color='blue')
    outliers_mask = df_original.iloc[:, 0] != df_tratado.iloc[:, 0]
    if outliers_mask.any():
        outliers_indices = df_original[outliers_mask].index
        plt.scatter(outliers_indices, df_original.loc[outliers_indices], color='red', s=100, marker='x', label='Outliers')
        plt.plot(df_tratado.index, df_tratado, 'o-', alpha=0.9, label='Série Tratada', color='green')
    plt.title(f'Detecção de Outliers - Região {region_name.upper()}', fontsize=16)
    plt.xlabel('Data', fontsize=12); plt.ylabel('Preço Médio (R$)', fontsize=12)
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(filepath); plt.close()
    print(f"Gráfico de outliers salvo em: {filepath}")

def plotar_decomposicao_stl(serie, region_name):
    output_dir = "static/images"
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    filepath = f"{output_dir}/decomposition_{region_name.lower()}.png"
    serie_resampled = serie.asfreq('MS')
    stl = STL(serie_resampled, seasonal=13)
    resultado = stl.fit()
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    resultado.observed.plot(ax=ax1, legend=False); ax1.set_ylabel('Observado')
    resultado.trend.plot(ax=ax2, legend=False); ax2.set_ylabel('Tendência')
    resultado.seasonal.plot(ax=ax3, legend=False); ax3.set_ylabel('Sazonalidade')
    resultado.resid.plot(ax=ax4, legend=False); ax4.set_ylabel('Resíduo')
    fig.suptitle(f'Decomposição da Série Temporal - Região {region_name.upper()}', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.savefig(filepath); plt.close()
    print(f"Gráfico de decomposição salvo em: {filepath}")

def realizar_previsao(df_region, future_dates, region):
    df_tratado_serie = tratar_outliers(df_region[region])
    df_tratado = pd.DataFrame(df_tratado_serie)
    df_prophet_input = df_tratado.reset_index().rename(columns={'MES': 'ds', region: 'y'})
    model_prophet = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False).fit(df_prophet_input)
    forecast_prophet = model_prophet.predict(pd.DataFrame({'ds': future_dates}))
    model_arima = ARIMA(df_tratado[region], order=(5, 1, 0)).fit()
    forecast_arima = model_arima.forecast(steps=len(future_dates))
    model_sarima = SARIMAX(df_tratado[region], order=(5, 1, 0), seasonal_order=(1, 1, 0, 12)).fit(disp=False)
    forecast_sarima = model_sarima.forecast(steps=len(future_dates))
    return forecast_prophet, forecast_arima, forecast_sarima, df_tratado

def plotar_previsoes_futuras(original, tratado, forecasts, future_dates, region):
    output_dir = "static/images"
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    filepath = f"{output_dir}/forecast_{region.lower()}.png"
    plt.figure(figsize=(14, 8))
    plt.plot(original.iloc[-36:].index, original.iloc[-36:][region], 'o-', label='Histórico (Original)', color='blue', alpha=0.4)
    plt.plot(tratado.iloc[-36:].index, tratado.iloc[-36:][region], 'o-', label='Histórico (Tratado)', color='darkblue')
    plt.plot(future_dates, forecasts['prophet']['yhat'], '--', label='Prophet', color='red')
    plt.plot(future_dates, forecasts['arima'], '--', label='ARIMA', color='green')
    plt.plot(future_dates, forecasts['sarima'], '--', label='SARIMA', color='purple')
    plt.fill_between(future_dates, forecasts['prophet']['yhat_lower'], forecasts['prophet']['yhat_upper'], color='red', alpha=0.2, label='IC Prophet 95%')
    plt.title(f'Previsões de Preço Médio de Energia - Região {region.upper()}', fontsize=16)
    plt.xlabel('Data', fontsize=12); plt.ylabel('Preço Médio (R$)', fontsize=12)
    plt.grid(True); plt.legend()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b/%Y'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.gcf().autofmt_xdate(); plt.tight_layout()
    plt.savefig(filepath); plt.close()
    print(f"Gráfico de previsões salvo em: {filepath}")

# --- NOVAS FUNÇÕES: Copiadas e adaptadas do Notebook para a Árvore de Decisão ---
def criar_arvore_decisao_alternativas(df_dist):
    output_dir = "static/images"
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    filepath = f"{output_dir}/decision_tree_alternatives.png"

    alternativas = {
        'Enfardamento': 80000,
        'Colheita integral': 70000,
        'Colheita parcial': 37500
    }
    
    receitas_esperadas = {}
    for nome, producao in alternativas.items():
        receita_total = 0
        for _, row in df_dist.iterrows():
            prob = row['Frequência'] / 100
            receita_total += prob * row['Ponto médio'] * producao
        receitas_esperadas[nome] = receita_total

    melhor_alternativa = max(receitas_esperadas, key=receitas_esperadas.get)
    maior_receita = receitas_esperadas[melhor_alternativa]

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    ax.plot([10, 30], [50, 50], 'k-')
    ax.text(5, 50, 'Escolha do método\nde recolhimento\ndo palhiço', ha='center', va='center', bbox=dict(boxstyle="square,pad=0.5", fc="white", ec="black"))

    pos_y = [75, 50, 25]
    for i, (nome, receita) in enumerate(receitas_esperadas.items()):
        ax.plot([30, 50], [50, pos_y[i]], 'k-')
        ax.text(55, pos_y[i], f'{nome}\nE(Receita): R$ {receita:,.2f}', ha='left', va='center', bbox=dict(boxstyle="round,pad=0.5", fc="lightblue", ec="black"))

    ax.set_title('Árvore de Decisão para Métodos de Recolhimento do Palhiço', fontsize=16)
    fig.text(0.5, 0.05, f'Melhor alternativa: {melhor_alternativa} com E(Receita): R$ {maior_receita:,.2f}', ha='center', fontsize=14, color='green', weight='bold')
    
    plt.savefig(filepath)
    plt.close()
    print(f"Gráfico da Árvore de Decisão salvo em: {filepath}")


# --- Função Principal Modificada ---
def main():
    print("Iniciando a geração de dados estáticos...")
    df_full = pd.read_excel("Historico_do_Preco_Medio_Mensal_-_janeiro_de_2001_a_abril_de_2025.xls")
    df_full['MES'] = pd.to_datetime(df_full['MES'], format='%b-%y', errors='coerce')
    
    # --- GERAÇÃO DA ÁRVORE DE DECISÃO (baseada nos dados históricos do Sudeste) ---
    print("\n--- Gerando Análise de Árvore de Decisão ---")
    df_sudeste = df_full[['MES', 'SUDESTE']].dropna()
    df_sudeste['SUDESTE'] = tratar_outliers(df_sudeste['SUDESTE'])
    int_cl = 100
    limites = np.arange(0, np.ceil(df_sudeste['SUDESTE'].max() / int_cl) * int_cl + int_cl, int_cl)
    classes = []
    for i in range(len(limites) - 1):
        li, ls = limites[i], limites[i + 1]
        ocorr = ((df_sudeste['SUDESTE'] >= li) & (df_sudeste['SUDESTE'] < ls)).sum()
        pm = (li + ls) / 2
        freq = round(ocorr / len(df_sudeste) * 100, 3)
        classes.append({'Classe': i + 1, 'Limite inferior': li, 'Limite superior': ls, 'Ponto médio': pm, 'Ocorrências': ocorr, 'Frequência': freq})
    df_dist = pd.DataFrame(classes)
    criar_arvore_decisao_alternativas(df_dist)


    # --- GERAÇÃO DOS DADOS DE PREVISÃO PARA CADA REGIÃO ---
    regioes = ['SUDESTE', 'SUL', 'NORDESTE', 'NORTE']
    future_dates = pd.date_range(start='2024-06-01', periods=16, freq='M')
    data_dir = "static/data"
    if not os.path.exists(data_dir): os.makedirs(data_dir)

    for region in regioes:
        print(f"\n--- Processando Região para Previsão: {region} ---")
        df_region = df_full[['MES', region]].dropna().set_index('MES')
        prophet, arima, sarima, df_tratado = realizar_previsao(df_region, future_dates, region)
        plotar_decomposicao_stl(df_tratado[region], region)
        forecasts = {'prophet': prophet, 'arima': arima, 'sarima': sarima}
        visualizar_outliers(df_region, pd.DataFrame(df_tratado[region]), region)
        plotar_previsoes_futuras(df_region, pd.DataFrame(df_tratado[region]), forecasts, future_dates, region)
        forecast_table_data = {'Data': future_dates.strftime('%b/%Y').tolist(), 'Prophet': prophet['yhat'].tolist(), 'ARIMA': arima.tolist(), 'SARIMA': sarima.tolist()}
        json_filepath = f"{data_dir}/forecast_data_{region.lower()}.json"
        with open(json_filepath, 'w', encoding='utf-8') as f:
            json.dump(forecast_table_data, f, ensure_ascii=False, indent=4)
        print(f"Dados da tabela salvos em: {json_filepath}")

    print("\n[SUCESSO] Geração de dados estáticos concluída!")

if __name__ == '__main__':
    main()