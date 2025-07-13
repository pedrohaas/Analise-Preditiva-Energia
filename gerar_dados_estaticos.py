import pandas as pd
import numpy as np
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import STL
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats
from matplotlib.table import Table
import os
import json
import warnings
import google.generativeai as genai

# --- CONFIGURAÇÕES GLOBAIS ---
warnings.filterwarnings('ignore')
plt.style.use('ggplot')
OUTPUT_DIR_IMAGES = "static/images"
OUTPUT_DIR_DATA = "static/data"

# --- FUNÇÕES DE ANÁLISE ---

def tratar_outliers(serie, metodo='iqr', limite=1.5):
    serie_limpa = serie.copy()
    q1, q3 = serie.quantile(0.25), serie.quantile(0.75)
    iqr = q3 - q1
    limite_inferior, limite_superior = q1 - limite * iqr, q3 + limite * iqr
    outliers_indices = np.where((serie < limite_inferior) | (serie > limite_superior))[0]
    janela = 5
    for idx in outliers_indices:
        inicio, fim = max(0, idx - janela//2), min(len(serie), idx + janela//2 + 1)
        vizinhanca = [val for i, val in enumerate(serie.iloc[inicio:fim]) if i + inicio != idx]
        if vizinhanca: serie_limpa.iloc[idx] = np.median(vizinhanca)
    return serie_limpa, len(outliers_indices)

def visualizar_outliers(df_original, df_tratado, region_name):
    """Gera e salva o gráfico de detecção de outliers."""
    filepath = f"{OUTPUT_DIR_IMAGES}/outliers_{region_name.lower()}.png"
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
    print(f"Imagem de outliers salva: {filepath}")

def plotar_decomposicao_stl(serie, region_name):
    """Gera e salva o gráfico de decomposição STL."""
    filepath = f"{OUTPUT_DIR_IMAGES}/decomposition_{region_name.lower()}.png"
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
    print(f"Imagem de decomposição salva: {filepath}")

def realizar_previsao(df_region, future_dates, region):
    """Realiza a previsão com os 3 modelos para o futuro."""
    df_tratado_serie, _ = tratar_outliers(df_region[region])
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
    """Gera e salva o gráfico com as previsões futuras."""
    filepath = f"{OUTPUT_DIR_IMAGES}/forecast_{region.lower()}.png"
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
    print(f"Imagem de previsões salva: {filepath}")

def criar_quadro_min_max(min_global, max_global, min_por_ano, max_por_ano):
    """Cria a imagem da tabela de preços mínimos e máximos."""
    filepath = f"{OUTPUT_DIR_IMAGES}/quadro_min_max.png"
    anos = ['Global'] + [str(a) for a in min_por_ano.index]
    minimos = [min_global] + list(min_por_ano.values)
    maximos = [max_global] + [max_por_ano[a] for a in min_por_ano.index]
    df_mm = pd.DataFrame({'Período': anos, 'Valor Mínimo (R$)': minimos, 'Valor Máximo (R$)': maximos})
    fig, ax = plt.subplots(figsize=(10, 0.2 * len(df_mm) + 1)); ax.axis('off'); ax.axis('tight')
    tabela = Table(ax, bbox=[0, 0, 1, 1]); larguras = [0.3, 0.35, 0.35]; altura_linha = 0.3
    tabela.auto_set_font_size(False); tabela.set_fontsize(10)
    tabela.add_cell(0, -1, width=sum(larguras), height=altura_linha*1.5, text='Valores mínimos e máximos do PLD (Sudeste/CO)', loc='center', facecolor='lightgray')
    for col_idx, col_name in enumerate(df_mm.columns):
        tabela.add_cell(1, col_idx, width=larguras[col_idx], height=altura_linha, text=col_name, loc='center', facecolor='lightyellow')
    for i in range(len(df_mm)):
        for j in range(len(df_mm.columns)):
            texto = df_mm.iloc[i, j]
            if isinstance(texto, (int, float)): texto = f'R$ {texto:,.2f}'
            tabela.add_cell(i+2, j, width=larguras[j], height=altura_linha, text=str(texto), loc='center')
    ax.add_table(tabela); plt.tight_layout(); plt.savefig(filepath, bbox_inches='tight', dpi=150); plt.close()
    print(f"Imagem do Quadro Min/Max salva: {filepath}")

def plotar_sensibilidade_receita():
    """Cria a imagem da análise de sensibilidade."""
    filepath = f"{OUTPUT_DIR_IMAGES}/sensibilidade_receita.png"
    precos = np.linspace(0, 800, 100); producao_integral = 70000; producao_enfardamento = 80000; producao_parcial = 37500
    receita_integral = precos * producao_integral; receita_enfardamento = precos * producao_enfardamento; receita_parcial = precos * producao_parcial
    plt.figure(figsize=(10,6)); plt.plot(precos, receita_integral, label='Colheita integral', linewidth=2)
    plt.plot(precos, receita_enfardamento, label='Enfardamento', linewidth=2)
    plt.plot(precos, receita_parcial, label='Colheita parcial', linewidth=2)
    plt.xlabel('Preço médio da energia (R$/MWh)'); plt.ylabel('Receita (R$)')
    plt.title('Análise de sensibilidade da receita por alternativa'); plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(filepath); plt.close()
    print(f"Imagem de Sensibilidade salva: {filepath}")

def criar_arvore_horizontal_classes(df_dist, forma_recolhimento, producao_excedente):
    """Cria a imagem de uma árvore de decisão individual."""
    filename = f"arvore_{forma_recolhimento.lower().replace(' ', '_')}.png"
    filepath = os.path.join(OUTPUT_DIR_IMAGES, filename)
    receita_esperada = (df_dist['Frequência']/100 * df_dist['Ponto médio'] * producao_excedente).sum()
    fig, ax = plt.subplots(figsize=(12, 8)); ax.axis('off')
    ax.plot([10, 30], [50, 50], 'k-')
    ax.text(10, 50, f'{forma_recolhimento}\nProdução: {producao_excedente:,} MWh', ha='center', va='center', bbox=dict(boxstyle="square,pad=0.5", fc="white", ec="black"))
    y_positions = np.linspace(95, 5, len(df_dist))
    for i, row in df_dist.iterrows():
        prob = row['Frequência']
        preco = row['Ponto médio']
        receita = preco * producao_excedente
        ax.plot([30, 50], [50, y_positions[i]], 'k-')
        ax.text(55, y_positions[i], f'Preço: R$ {preco:,.2f} ({prob}%)\nE(Receita): R$ {receita:,.2f}', ha='left', va='center', bbox=dict(boxstyle="round,pad=0.5", fc="lightblue", ec="black"))
    ax.set_title(f'Árvore de Decisão para {forma_recolhimento}', fontsize=16)
    fig.text(0.5, 0.02, f'Valor Esperado Total (Receita): R$ {receita_esperada:,.2f}', ha='center', fontsize=14, color='green', weight='bold')
    plt.savefig(filepath, bbox_inches='tight'); plt.close()
    print(f"Imagem da Árvore '{forma_recolhimento}' salva: {filepath}")

def criar_arvore_decisao_alternativas(df_dist):
    """Cria a imagem da árvore de decisão final comparativa."""
    filepath = f"{OUTPUT_DIR_IMAGES}/decision_tree_alternatives.png"
    alternativas = {'Enfardamento': 80000, 'Colheita integral': 70000, 'Colheita parcial': 37500}
    receitas_esperadas = {nome: (df_dist['Frequência']/100 * df_dist['Ponto médio'] * producao).sum() for nome, producao in alternativas.items()}
    melhor_alternativa = max(receitas_esperadas, key=receitas_esperadas.get)
    maior_receita = receitas_esperadas[melhor_alternativa]
    fig, ax = plt.subplots(figsize=(14, 8)); ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis('off')
    ax.plot([10, 30], [50, 50], 'k-')
    ax.text(5, 50, 'Escolha do método\nde recolhimento', ha='center', va='center', bbox=dict(boxstyle="square,pad=0.5", fc="white", ec="black"))
    pos_y = [75, 50, 25]
    for i, (nome, receita) in enumerate(receitas_esperadas.items()):
        ax.plot([30, 50], [50, pos_y[i]], 'k-')
        ax.text(55, pos_y[i], f'{nome}\nE(Receita): R$ {receita:,.2f}', ha='left', va='center', bbox=dict(boxstyle="round,pad=0.5", fc="lightblue", ec="black"))
    ax.set_title('Árvore de Decisão Comparativa dos Métodos', fontsize=16)
    fig.text(0.5, 0.05, f'Melhor alternativa: {melhor_alternativa} com E(Receita): R$ {maior_receita:,.2f}', ha='center', fontsize=14, color='green', weight='bold')
    plt.savefig(filepath); plt.close()
    print(f"Imagem da Árvore Comparativa salva: {filepath}")

def gerar_insights_com_ia(dados_validacao, dados_arvore, dados_previsao_futura):
    """
    Consolida TODAS as análises, cria um prompt estratégico e chama a API do Gemini.
    """
    print("\n--- Gerando Insights Estratégicos com a IA do Gemini ---")
    
    try:
        # Configuração da API
        GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
        if not GOOGLE_API_KEY:
            print("AVISO: Chave de API do Google não encontrada. A geração de insights com IA será ignorada.")
            return

        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel('gemini-2.5-pro')

        # --- CONTEXTO HOLÍSTICO: Fornecer um resumo de todas as análises ---
        sudeste_metrics = dados_validacao['sudeste']['metrics']
        
        def format_metric(model_name, metric_name):
            value = sudeste_metrics.get(model_name, {}).get(metric_name)
            return f"{value:.2f}" if value is not None else "N/A"

        contexto = f"""
        Resumo das Análises para Tomada de Decisão Estratégica na Geração de Energia:

        1. ANÁLISE HISTÓRICA (Baseada em probabilidades passadas):
           - Análise da Árvore de Decisão para métodos de recolhimento de palhiço de cana.
           - Melhor alternativa histórica: '{dados_arvore['melhor_alternativa']}', com um Valor Monetário Esperado de R$ {dados_arvore['maior_receita']:,.2f}.

        2. ANÁLISE PREDITIVA (Baseada em previsões para os próximos 16 meses na região Sudeste):
           - Validação dos Modelos: O modelo SARIMA apresentou o melhor equilíbrio, com MAE de {format_metric('SARIMA', 'MAE')} e RMSE de {format_metric('SARIMA', 'RMSE')}, sendo o mais recomendado por capturar bem a sazonalidade.
           - Previsão Futura (Modelo SARIMA): A média dos preços previstos para os próximos 16 meses é de R$ {dados_previsao_futura['media_preco_sarima']:.2f}/MWh.

        3. SIMULAÇÃO DE INVESTIMENTO (Payback da Usina de Palhiço):
           - Custo do Investimento (exemplo): R$ 310.000
           - Usando o preço médio futuro previsto pelo SARIMA (R$ {dados_previsao_futura['media_preco_sarima']:.2f}), o tempo de retorno (payback) do investimento seria de aproximadamente {dados_previsao_futura['payback_meses']} meses para um consumo de 540 MWh/mês.
        """

        # --- PROMPT ESTRATÉGICO FINAL ---
        prompt = f"""
        Você é um consultor sênior de energia e estratégia, preparando um sumário executivo final para a diretoria de uma usina.
        Seu papel é sintetizar todas as análises (histórica, preditiva e de simulação) em recomendações estratégicas de alto nível.
        Vá além da simples escolha de um modelo. Fale sobre oportunidades, riscos e como as diferentes análises se conectam.
        
        Com base no contexto completo a seguir, escreva um relatório conciso em 4 tópicos.
        **Use as tags <strong> e </strong> para destacar os termos mais importantes.**

        Contexto da Análise:
        {contexto}

        Forneça sua resposta estritamente no formato JSON, com uma chave "titulo" e uma chave "insights" que é uma lista de 4 strings (contendo as tags <strong>).
        """

        safety_config = {
            'HARM_CATEGORY_HARASSMENT': 'BLOCK_NONE', 'HARM_CATEGORY_HATE_SPEECH': 'BLOCK_NONE',
            'HARM_CATEGORY_SEXUALLY_EXPLICIT': 'BLOCK_NONE', 'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_NONE',
        }
        
        response = model.generate_content(prompt, safety_settings=safety_config)
        
        cleaned_response_text = response.text.strip().replace('```json', '').replace('```', '').strip()
        
        insights_data = json.loads(cleaned_response_text)
        filepath = os.path.join(OUTPUT_DIR_DATA, "insights_ia.json")
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(insights_data, f, ensure_ascii=False, indent=4)
        
        print(f"Insights estratégicos da IA salvos com sucesso em: {filepath}")

    except Exception as e:
        print(f"\nERRO ao gerar insights com a IA: {e}")



# --- FUNÇÃO PRINCIPAL ---
def main():
    """Orquestra a geração de todos os ativos estáticos para o site."""
    print("Iniciando a geração de todos os dados e gráficos estáticos...")
    
    # Garante que os diretórios de saída existam
    os.makedirs(OUTPUT_DIR_IMAGES, exist_ok=True)
    os.makedirs(OUTPUT_DIR_DATA, exist_ok=True)
    
    # Carregamento dos dados
    try:
        df_full = pd.read_excel("Historico_do_Preco_Medio_Mensal_-_janeiro_de_2001_a_abril_de_2025.xls")
        df_full['MES'] = pd.to_datetime(df_full['MES'], format='%b-%y', errors='coerce')
    except FileNotFoundError:
        print("ERRO: Arquivo 'Historico_do_Preco_Medio_Mensal_-_janeiro_de_2001_a_abril_de_2025.xls' não encontrado.")
        return

    # --- Bloco 1: GERAÇÃO DAS ANÁLISES DE DECISÃO (baseadas nos dados históricos do Sudeste) ---
    print("\n--- Gerando Análises de Decisão e Sensibilidade ---")
    df_sudeste_original = df_full[['MES', 'SUDESTE']].dropna()
    df_sudeste_serie, _ = tratar_outliers(df_sudeste_original['SUDESTE'])
    df_sudeste = pd.DataFrame(df_sudeste_serie)

    min_g = df_sudeste_original['SUDESTE'].min()
    max_g = df_sudeste_original['SUDESTE'].max()
    min_ano = df_sudeste_original.groupby(df_sudeste_original['MES'].dt.year)['SUDESTE'].min()
    max_ano = df_sudeste_original.groupby(df_sudeste_original['MES'].dt.year)['SUDESTE'].max()
    criar_quadro_min_max(min_g, max_g, min_ano, max_ano)

    plotar_sensibilidade_receita()

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
    
    criar_arvore_horizontal_classes(df_dist,  forma_recolhimento='Colheita integral', producao_excedente=70000)
    criar_arvore_horizontal_classes(df_dist, forma_recolhimento='Enfardamento', producao_excedente=80000)
    criar_arvore_horizontal_classes(df_dist, forma_recolhimento='Colheita parcial', producao_excedente=37500)
    
    # Coleta os dados da árvore de decisão para a IA
    alternativas_arvore = {'Enfardamento': 80000, 'Colheita integral': 70000, 'Colheita parcial': 37500}
    receitas_esperadas = {nome: (df_dist['Frequência']/100 * df_dist['Ponto médio'] * producao).sum() for nome, producao in alternativas_arvore.items()}
    melhor_alternativa_arvore = max(receitas_esperadas, key=receitas_esperadas.get)
    maior_receita_arvore = receitas_esperadas[melhor_alternativa_arvore]
    dados_arvore_para_ia = {"melhor_alternativa": melhor_alternativa_arvore, "maior_receita": maior_receita_arvore}
    criar_arvore_decisao_alternativas(df_dist)


    # --- Bloco 2: GERAÇÃO DOS DADOS DE PREVISÃO E VALIDAÇÃO ---
    regioes = ['SUDESTE', 'SUL', 'NORDESTE', 'NORTE']
    future_dates = pd.date_range(start='2024-06-01', periods=16, freq='M')
    all_validation_metrics = {}
    dados_previsao_futura_para_ia = {} # Dicionário para guardar os dados para a IA

    for region in regioes:
        print(f"\n--- Processando Região para Previsão e Validação: {region} ---")
        df_region = df_full[['MES', region]].dropna().set_index('MES')
        
        df_tratado_serie, num_outliers = tratar_outliers(df_region[region])
        df_tratado = pd.DataFrame(df_tratado_serie)
        
        train_data, test_data = df_tratado[:-5], df_tratado[-5:]
        
        prophet_val = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False).fit(train_data.reset_index().rename(columns={'MES': 'ds', region: 'y'}))
        prophet_pred = prophet_val.predict(test_data.reset_index().rename(columns={'MES': 'ds'}))['yhat'].values
        arima_val = ARIMA(train_data[region], order=(5,1,0)).fit()
        arima_pred = arima_val.forecast(steps=5)
        sarima_val = SARIMAX(train_data[region], order=(5,1,0), seasonal_order=(1,1,0,12)).fit(disp=False)
        sarima_pred = sarima_val.forecast(steps=5)
        
        try:
            stl_result = STL(train_data[region].asfreq('MS'), seasonal=13).fit()
            stl_pred = stl_result.forecast(5)
        except AttributeError:
            print(f"AVISO: A versão do statsmodels instalada não suporta 'STL.forecast()'. A validação para o modelo STL será ignorada.")
            stl_pred = np.full(5, np.nan)

        def calculate_metrics(y_true, y_pred):
            if np.isnan(y_pred).any(): return {'MAE': None, 'MSE': None, 'RMSE': None}
            return {'MAE': mean_absolute_error(y_true, y_pred), 'MSE': mean_squared_error(y_true, y_pred), 'RMSE': np.sqrt(mean_squared_error(y_true, y_pred))}

        metrics = {
            'Prophet': calculate_metrics(test_data[region], prophet_pred), 'ARIMA': calculate_metrics(test_data[region], arima_pred),
            'SARIMA': calculate_metrics(test_data[region], sarima_pred), 'STL': calculate_metrics(test_data[region], stl_pred),
        }
        all_validation_metrics[region.lower()] = {'outliers_count': num_outliers, 'outliers_percentage': round((num_outliers / len(df_region)) * 100, 2), 'metrics': metrics}
        
        prophet_full, arima_full, sarima_full, df_tratado_full = realizar_previsao(df_region, future_dates, region)
        
        # Coleta os dados de previsão futura para a IA (apenas para a região de referência, Sudeste)
        if region == 'SUDESTE':
            media_preco_sarima = np.mean(sarima_full)
            custo_investimento_exemplo = 310000
            consumo_mensal_exemplo = 540
            payback_meses_exemplo = custo_investimento_exemplo / (media_preco_sarima * consumo_mensal_exemplo)
            dados_previsao_futura_para_ia = {'media_preco_sarima': media_preco_sarima, 'payback_meses': payback_meses_exemplo}

        plotar_decomposicao_stl(df_tratado_full[region], region)
        forecasts = {'prophet': prophet_full, 'arima': arima_full, 'sarima': sarima_full}
        visualizar_outliers(df_region, df_tratado_full, region)
        plotar_previsoes_futuras(df_region, df_tratado_full, forecasts, future_dates, region)
        
        forecast_table_data = {'Data': future_dates.strftime('%b/%Y').tolist(), 'Prophet': prophet_full['yhat'].tolist(), 'ARIMA': arima_full.tolist(), 'SARIMA': sarima_full.tolist()}
        json_filepath = f"{OUTPUT_DIR_DATA}/forecast_data_{region.lower()}.json"
        with open(json_filepath, 'w', encoding='utf-8') as f:
            json.dump(forecast_table_data, f, ensure_ascii=False, indent=4)
        print(f"Dados de previsão salvos: {json_filepath}")

    validation_filepath = os.path.join(OUTPUT_DIR_DATA, "validation_metrics.json")
    with open(validation_filepath, 'w', encoding='utf-8') as f:
        json.dump(all_validation_metrics, f, ensure_ascii=False, indent=4)
    print(f"\nArquivo com métricas de validação salvo: {validation_filepath}")
    
    # --- Bloco 3: CHAMADA FINAL PARA A IA COM TODOS OS DADOS CONSOLIDADOS ---
    gerar_insights_com_ia(all_validation_metrics, dados_arvore_para_ia, dados_previsao_futura_para_ia)

    print("\n[SUCESSO] Geração de todos os dados estáticos concluída!")
    
if __name__ == '__main__':
    main()