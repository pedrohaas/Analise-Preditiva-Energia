# Dashboard de Análise Preditiva para o Mercado de Energia

![Licença](https://img.shields.io/badge/license-MIT-blue.svg)
![Status](https://img.shields.io/badge/status-concluído-green.svg)

Um dashboard web interativo e responsivo para análise de séries temporais, previsão de preços de energia e suporte à decisão estratégica, com insights gerados por Inteligência Artificial.

**[➡️ Acessar a Demonstração Ao Vivo](https://pedrohaas.github.io/T1-TomadaDecisao/)**

---

## Tabela de Conteúdos

- [Sobre o Projeto](#sobre-o-projeto)
  - [Metodologia Analítica](#metodologia-analítica)
- [Principais Funcionalidades](#principais-funcionalidades)
- [Pilha de Tecnologias](#pilha-de-tecnologias)
- [Como Executar o Projeto Localmente](#como-executar-o-projeto-localmente)
  - [Pré-requisitos](#pré-requisitos)
  - [Instalação](#instalação)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Licença](#licença)
- [Autores](#autores)

---

## Sobre o Projeto

Este projeto nasceu de um desafio acadêmico de "Tomada de Decisão em Contextos Complexos" e evoluiu para uma aplicação web completa. O objetivo é fornecer uma ferramenta de suporte à decisão para uma usina de açúcar e álcool, ajudando a responder a uma questão de negócio central: **qual a melhor estratégia para o recolhimento do palhiço de cana, considerando seu potencial para geração e venda de energia?**

Para responder a essa pergunta, o dashboard integra duas abordagens analíticas distintas e complementares.

### Metodologia Analítica

1.  **Análise de Decisão Histórica:** Utiliza uma **Árvore de Decisão** clássica, baseada na distribuição de probabilidade dos preços históricos de energia da região Sudeste. Esta análise calcula o Valor Monetário Esperado (VME) para cada método de recolhimento, identificando a estratégia mais lucrativa se o futuro se comportar exatamente como o passado.

2.  **Análise de Cenários Futuros (Preditiva):** Emprega um conjunto de modelos de previsão de séries temporais (Prophet, ARIMA, SARIMA, STL) para estimar os preços de energia para os próximos 16 meses. Esses modelos são validados através de backtesting e suas previsões alimentam uma ferramenta interativa que calcula o retorno sobre o investimento (Payback) de uma nova usina, oferecendo uma visão de viabilidade financeira orientada para o futuro.

3.  **Inteligência Aumentada (IA):** O projeto culmina com a utilização da API do **Google Gemini** para sintetizar todas as análises (histórica, preditiva e de simulação) em um sumário executivo com insights estratégicos, agindo como um consultor automatizado.

---

## Principais Funcionalidades

- ✅ **Dashboard Interativo:** Interface web responsiva, elegante e intuitiva.
- ✅ **Análise Multi-Região:** Permite selecionar e analisar dados de preços das regiões Sudeste, Sul, Nordeste e Norte do Brasil.
- ✅ **Visualizações de Dados Detalhadas:** Gráficos para decomposição da série, tratamento de outliers e previsões comparativas.
- ✅ **Validação de Modelos:** Tabela transparente com métricas de erro (MAE, MSE, RMSE) para cada modelo, validando sua performance.
- ✅ **Galeria de Análise de Decisão:** Apresenta todos os gráficos de suporte da análise de árvore de decisão, incluindo análise de sensibilidade.
- ✅ **Ferramenta de Simulação de Investimento:** Calcula o retorno (Payback) de um investimento com base em parâmetros customizáveis pelo usuário.
- ✅ **Insights Gerados por IA:** Um sumário executivo gerado pela API do Gemini que oferece recomendações estratégicas de alto nível.
- ✅ **Visualizador de Imagens (Lightbox):** Permite expandir todos os gráficos para uma análise detalhada.

---

## Pilha de Tecnologias

Este projeto utiliza uma arquitetura de site estático, onde a análise pesada é pré-processada em Python e os resultados são consumidos por uma interface rica em JavaScript.

- **Análise de Dados e Geração Estática:**
  - Python
  - Pandas & NumPy
  - Matplotlib
  - Statsmodels (ARIMA, SARIMA, STL)
  - Prophet (da Meta)
  - Scikit-learn (para métricas de erro)
  - Google Generative AI (para integração com a API do Gemini)

- **Frontend:**
  - HTML5
  - CSS3 (com CSS Grid e Flexbox para responsividade)
  - JavaScript (Vanilla JS, sem frameworks)

---

## Como Executar o Projeto Localmente

Siga estas instruções para configurar e executar o projeto em sua máquina.

### Pré-requisitos

- Python (versão 3.8 ou superior)
- `pip` (gerenciador de pacotes do Python)

### Instalação

1.  **Clone o Repositório**
    ```sh
    git clone [https://github.com/pedrohaas/T1-TomadaDecisao.git](https://github.com/pedrohaas/T1-TomadaDecisao.git)
    cd T1-TomadaDecisao
    ```

2.  **Crie e Ative um Ambiente Virtual (Recomendado)**
    ```sh
    python -m venv venv
    # No Windows:
    venv\Scripts\activate
    # No macOS/Linux:
    source venv/bin/activate
    ```

3.  **Instale as Dependências**
    ```sh
    pip install -r requirements.txt
    ```

4.  **Configure sua Chave de API do Gemini**
    - Gere sua chave de API gratuita no [Google AI Studio](https://aistudio.google.com/app/apikey).
    - Configure-a como uma variável de ambiente. **Não cole a chave diretamente no código.**
      - No Windows (Prompt de Comando):
        ```sh
        setx GOOGLE_API_KEY "SUA_CHAVE_DE_API_AQUI"
        ```
        (Lembre-se de fechar e reabrir o terminal após executar este comando).
      - No macOS/Linux:
        ```sh
        export GOOGLE_API_KEY="SUA_CHAVE_DE_API_AQUI"
        ```

5.  **Gere os Ativos Estáticos**
    - Certifique-se de que o arquivo `Historico_do_Preco_Medio_Mensal_-_janeiro_de_2001_a_abril_de_2025.xls` esteja na pasta raiz do projeto.
    - Execute o script principal. Ele irá criar todas as imagens e arquivos de dados necessários.
    ```sh
    python gerar_dados_estaticos.py
    ```
    Este processo pode levar alguns minutos.

6.  **Visualize o Projeto**
    - Após a execução do script, simplesmente abra o arquivo `index.html` em seu navegador de preferência.

---

## Estrutura do Projeto

```
/
├── index.html                  # O arquivo principal da interface web.
├── gerar_dados_estaticos.py    # O "motor" do projeto: script Python que gera todos os dados e gráficos.
├── requirements.txt            # Lista de dependências Python.
├── LICENSE                     # A licença do projeto (MIT).
├── README.md                   # Esta documentação.
│
└── static/
    ├── style.css               # Folha de estilos para a interface.
    ├── script.js               # Lógica de interatividade do frontend.
    │
    ├── data/                   # Contém os arquivos de dados gerados.
    │   ├── forecast_data_*.json
    │   ├── validation_metrics.json
    │   └── insights_ia.json
    │
    └── images/                 # Contém todas as imagens e gráficos gerados.
        └── ...
```

---

## Licença

Este projeto está licenciado sob a Licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.

---

## Autores

- **Pedro Henrique Alves de Araujo Silva** - [LinkedIn](https://www.linkedin.com/in/opedroalves/)
- **Laura Rieko Marçal Imai** - [LinkedIn](https://www.linkedin.com/in/laura-rieko-imai/)
```