import pandas as pd
import numpy as np
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats
import os
import json
import warnings

warnings.filterwarnings('ignore')

# --- Funções de Análise ---

def tratar_outliers(serie, metodo='iqr', limite=1.5):
    serie_limpa = serie.copy()
    if metodo == 'iqr':
        Q1 = serie.quantile(0.25)
        Q3 = serie.quantile(0.75)
        IQR = Q3 - Q1
        outliers_indices = np.where((serie < (Q1 - limite * IQR)) | (serie > (Q3 + limite * IQR)))[0]
    else:
        z_scores = np.abs(stats.zscore(serie))
        outliers_indices = np.where(z_scores > limite)[0]
    
    janela = 5
    for idx in outliers_indices:
        inicio = max(0, idx - janela//2)
        fim = min(len(serie), idx + janela//2 + 1)
        vizinhanca = [val for i, val in enumerate(serie.iloc[inicio:fim]) if i + inicio != idx]
        if vizinhanca:
            serie_limpa.iloc[idx] = np.median(vizinhanca)
    return serie_limpa

def visualizar_outliers(df_original, df_tratado, region_name):
    output_dir = "static/images"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    filepath = f"{output_dir}/outliers_{region_name.lower()}.png"
    plt.figure(figsize=(14, 6))
    plt.plot(df_original.index, df_original, 'o-', alpha=0.7, label='Série Original', color='blue')
    serie_original = df_original.iloc[:, 0]
    serie_tratada = df_tratado.iloc[:, 0]
    outliers_mask = serie_original != serie_tratada
    if outliers_mask.any():
        outliers_indices = outliers_mask[outliers_mask].index
        plt.scatter(outliers_indices, serie_original[outliers_indices],
                   color='red', s=100, marker='x', label='Outliers')
        plt.plot(df_tratado.index, df_tratado, 'o-', alpha=0.9,
                 label='Série Tratada', color='green')
    plt.title(f'Detecção de Outliers - {region_name}', fontsize=16)
    plt.xlabel('Data', fontsize=12)
    plt.ylabel('Preço Médio (R$)', fontsize=12)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()
    print(f"Gráfico de outliers salvo em: {filepath}")

# CORREÇÃO 1: Adicionar 'region' como parâmetro da função
def realizar_previsao(df_region, future_dates, region):
    # O nome da coluna do DataFrame original é a própria região (ex: 'SUDESTE')
    # Extraímos essa coluna para o tratamento de outliers
    df_region_tratado = pd.DataFrame(tratar_outliers(df_region[region]))
    df_region_tratado.index = df_region.index

    # Prepara o DataFrame para o Prophet, renomeando as colunas
    df_prophet_input = df_region_tratado.reset_index().rename(columns={'MES': 'ds', region: 'y'})

    # Modelos
    model_prophet = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    model_prophet.fit(df_prophet_input)
    forecast_prophet = model_prophet.predict(pd.DataFrame({'ds': future_dates}))

    model_arima = ARIMA(df_region_tratado[region], order=(5, 1, 0))
    forecast_arima = model_arima.fit().forecast(steps=len(future_dates))

    model_sarima = SARIMAX(df_region_tratado[region], order=(5, 1, 0), seasonal_order=(1, 1, 0, 12))
    forecast_sarima = model_sarima.fit(disp=False).forecast(steps=len(future_dates))
    
    return forecast_prophet, forecast_arima, forecast_sarima, df_region_tratado

def plotar_previsoes_futuras(original, tratado, forecasts, future_dates, region):
    output_dir = "static/images"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    filepath = f"{output_dir}/forecast_{region.lower()}.png"
    plt.figure(figsize=(14, 8))
    plt.plot(original.iloc[-36:].index, original.iloc[-36:][region], 'o-', label='Histórico (Original)', color='blue', alpha=0.4)
    plt.plot(tratado.iloc[-36:].index, tratado.iloc[-36:][region], 'o-', label='Histórico (Tratado)', color='darkblue')
    
    plt.plot(future_dates, forecasts['prophet']['yhat'], '--', label='Prophet', color='red')
    plt.plot(future_dates, forecasts['arima'], '--', label='ARIMA', color='green')
    plt.plot(future_dates, forecasts['sarima'], '--', label='SARIMA', color='purple')
    
    plt.fill_between(future_dates, forecasts['prophet']['yhat_lower'], forecasts['prophet']['yhat_upper'], color='red', alpha=0.2, label='IC Prophet 95%')
    
    plt.title(f'Previsões de Preço Médio de Energia - Região {region}', fontsize=16)
    plt.xlabel('Data', fontsize=12)
    plt.ylabel('Preço Médio (R$)', fontsize=12)
    plt.grid(True)
    plt.legend()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b/%Y'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()
    print(f"Gráfico de previsões salvo em: {filepath}")

def main():
    print("Iniciando a geração de dados estáticos...")
    df = pd.read_excel("Historico_do_Preco_Medio_Mensal_-_janeiro_de_2001_a_abril_de_2025.xls")
    df['MES'] = pd.to_datetime(df['MES'], format='%b-%y', errors='coerce')
    
    region = 'SUDESTE'
    df_region = df[['MES', region]].dropna().set_index('MES')
    
    future_dates = pd.date_range(start='2024-06-01', periods=16, freq='M')

    # CORREÇÃO 2: Passar 'region' como argumento na chamada da função
    prophet, arima, sarima, df_tratado = realizar_previsao(df_region, future_dates, region)
    
    forecasts = {'prophet': prophet, 'arima': arima, 'sarima': sarima}

    visualizar_outliers(df_region, df_tratado, region)
    plotar_previsoes_futuras(df_region, df_tratado, forecasts, future_dates, region)

    forecast_table_data = {
        'Data': future_dates.strftime('%b/%Y').tolist(),
        'Prophet': [f'R$ {x:.2f}' for x in prophet['yhat']],
        'ARIMA': [f'R$ {x:.2f}' for x in arima],
        'SARIMA': [f'R$ {x:.2f}' for x in sarima]
    }
    
    data_dir = "static/data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    json_filepath = f"{data_dir}/forecast_data.json"
    with open(json_filepath, 'w', encoding='utf-8') as f:
        json.dump(forecast_table_data, f, ensure_ascii=False, indent=4)
    
    print(f"Dados da tabela salvos em: {json_filepath}")
    print("\nGeração de dados estáticos concluída!")

if __name__ == '__main__':
    main()