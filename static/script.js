document.addEventListener('DOMContentLoaded', function() {

    // --- ELEMENTOS GLOBAIS DO DOM ---
    const regionButtons = document.querySelectorAll('.region-btn');
    const investmentCostInput = document.getElementById('investmentCost');
    const energyConsumptionInput = document.getElementById('energyConsumption');
    const loaderOverlay = document.getElementById('loader-overlay');

    // --- ARMAZENAMENTO DE DADOS EM CACHE ---
    let currentForecastData = {};
    let allValidationMetrics = {}; // Carregado apenas uma vez para otimização

    // --- FUNÇÃO PRINCIPAL QUE ORQUESTRA TODAS AS ATUALIZAÇÕES ---
    async function updateDashboard(region) {
        showLoader();

        if (Object.keys(allValidationMetrics).length === 0) { // Verifica se é a primeira carga
        await Promise.all([
            loadForecastData(region),
            loadValidationData(),
            loadAndRenderAIInsights() // Carrega os insights da IA junto
        ]);
        } else {
            await loadForecastData(region); // Em outras cargas, só carrega os dados da região
        }

        // Renderiza todas as seções do dashboard com os novos dados
        updateTitles(region);
        updateCharts(region);
        renderValidationSection(region);
        renderForecastTable();
        updateDecisionAnalysis();
        updateActiveButton(region);

        hideLoader();
    }

    // --- FUNÇÕES DE CARREGAMENTO DE DADOS (DATA FETCHING) ---

    async function loadForecastData(region) {
        const dataFile = `static/data/forecast_data_${region}.json`;
        try {
            const response = await fetch(dataFile);
            if (!response.ok) throw new Error(`Arquivo de previsão não encontrado para a região ${region}`);
            currentForecastData = await response.json();
        } catch (error) {
            console.error('Erro ao carregar dados de previsão:', error);
            currentForecastData = {}; // Limpa os dados em caso de erro para evitar falhas
        }
    }

    async function loadValidationData() {
        // Só carrega o arquivo de validação se ele ainda não foi carregado
        if (Object.keys(allValidationMetrics).length > 0) return;
        try {
            const response = await fetch('static/data/validation_metrics.json');
            if (!response.ok) throw new Error('Arquivo de métricas não encontrado');
            allValidationMetrics = await response.json();
        } catch (error) {
            console.error('Erro ao carregar métricas de validação:', error);
        }
    }

    // --- FUNÇÕES DE RENDERIZAÇÃO (ATUALIZAÇÃO DA INTERFACE) ---

    function updateTitles(region) {
        const regionNameCapitalized = capitalize(region);
        document.getElementById('outliers-title').textContent = `Passo 2: Preparação dos Dados (${regionNameCapitalized})`;
        document.getElementById('decomposition-card-title').textContent = `Decomposição da Série Temporal - ${regionNameCapitalized}`;
    }

    function updateCharts(region) {
        const cacheBuster = `?t=${new Date().getTime()}`; // Evita que o navegador use imagens antigas
        document.getElementById('decompositionChart').src = `static/images/decomposition_${region}.png${cacheBuster}`;
        document.getElementById('outliersChart').src = `static/images/outliers_${region}.png${cacheBuster}`;
        document.getElementById('forecastChart').src = `static/images/forecast_${region}.png${cacheBuster}`;
    }
    
    function renderValidationSection(region) {
        const regionMetricsData = allValidationMetrics[region];
        if (!regionMetricsData) return;

        // --- Parte 1: Atualizar o sumário de outliers (inalterado) ---
        const summaryEl = document.getElementById('validation-summary');
        summaryEl.innerHTML = `Para a região <strong>${capitalize(region)}</strong>, foram detectados e tratados <strong>${regionMetricsData.outliers_count} outliers</strong> (${regionMetricsData.outliers_percentage}% do total de dados).`;

        // --- Parte 2: Popular a tabela de métricas (inalterado) ---
        const tableBody = document.querySelector("#validation-table tbody");
        tableBody.innerHTML = '';
        
        const modelsMetrics = regionMetricsData.metrics;
        for (const modelName in modelsMetrics) {
            const metrics = modelsMetrics[modelName];
            const row = tableBody.insertRow();
            row.insertCell(0).textContent = modelName;
            row.insertCell(1).textContent = metrics.MAE !== null ? metrics.MAE.toFixed(2) : 'N/A';
            row.insertCell(2).textContent = metrics.MSE !== null ? metrics.MSE.toFixed(2) : 'N/A';
            row.insertCell(3).textContent = metrics.RMSE !== null ? metrics.RMSE.toFixed(2) : 'N/A';
        }

        // --- Parte 3: Lógica para encontrar e destacar o melhor modelo ---
        let bestModelName = null;
        let bestModelExplanation = "Não foi possível determinar um modelo com desempenho significativamente superior com base nas métricas disponíveis.";

        // Inicialmente, vamos considerar o SARIMA como um forte candidato devido à sua capacidade de lidar com sazonalidade
        if (modelsMetrics.SARIMA && modelsMetrics.ARIMA) {
            const sarimaRmse = modelsMetrics.SARIMA.RMSE;
            const sarimaMae = modelsMetrics.SARIMA.MAE;
            const arimaRmse = modelsMetrics.ARIMA.RMSE;
            const arimaMae = modelsMetrics.ARIMA.MAE;

            // Critérios mais balanceados: SARIMA tem RMSE razoavelmente bom e MAE competitivo com ARIMA
            if (sarimaRmse !== null && sarimaMae !== null && arimaRmse !== null && arimaMae !== null) {
                if (sarimaRmse <= arimaRmse + (0.1 * arimaRmse) && sarimaMae <= arimaMae + (0.05 * arimaMae)) {
                    bestModelName = "SARIMA";
                    bestModelExplanation = `Com base na validação, o modelo <strong>SARIMA</strong> demonstra uma excelente capacidade de capturar os padrões sazonais dos preços, apresentando um erro RMSE de <strong>${sarimaRmse.toFixed(2)}</strong> e um MAE de <strong>${sarimaMae.toFixed(2)}</strong>, sendo nossa principal referência para as previsões futuras.`;
                }
            }
        }

        // Se o SARIMA não atender aos critérios, podemos voltar para o modelo com o menor RMSE (Prophet) como padrão
        if (!bestModelName && modelsMetrics.Prophet) {
            let lowestRmse = Infinity;
            let prophetRmse = modelsMetrics.Prophet.RMSE;
            let prophetMae = modelsMetrics.Prophet.MAE;

            if (prophetRmse !== null && prophetRmse < lowestRmse) {
                lowestRmse = prophetRmse;
                bestModelName = "Prophet";
                bestModelExplanation = `Embora o modelo <strong>Prophet</strong> apresente o menor erro RMSE (<strong>${prophetRmse.toFixed(2)}</strong>), é importante notar que seu MAE (<strong>${prophetMae.toFixed(2)}</strong>) é maior que outros modelos. Ele ainda é uma referência importante, especialmente para a tendência geral dos preços.`;
            }
        }

        // Atualiza o texto no card de destaque no Passo 4
        const bestModelTextEl = document.getElementById('best-model-text');
        bestModelTextEl.innerHTML = bestModelExplanation;
    }

    function renderForecastTable() {
        const table = document.getElementById('forecastTable');
        table.innerHTML = '';
        if (!currentForecastData.Data) return;

        let thead = table.createTHead();
        let row = thead.insertRow();
        Object.keys(currentForecastData).forEach(key => {
            let th = document.createElement("th");
            th.innerHTML = key;
            row.appendChild(th);
        });

        let tbody = table.createTBody();
        for (let i = 0; i < currentForecastData.Data.length; i++) {
            let row = tbody.insertRow();
            row.insertCell(0).innerHTML = currentForecastData.Data[i];
            row.insertCell(1).innerHTML = formatCurrency(currentForecastData.Prophet[i]);
            row.insertCell(2).innerHTML = formatCurrency(currentForecastData.ARIMA[i]);
            row.insertCell(3).innerHTML = formatCurrency(currentForecastData.SARIMA[i]);
        }
    }

    function updateDecisionAnalysis() {
        if (!currentForecastData.Data) return;

        const investmentCost = parseFloat(investmentCostInput.value) || 0;
        const energyConsumption = parseFloat(energyConsumptionInput.value) || 0;
        const decisionTableBody = document.querySelector("#decisionTable tbody");
        decisionTableBody.innerHTML = '';

        const models = ['Prophet', 'ARIMA', 'SARIMA'];

        models.forEach(model => {
            const totalSavings = currentForecastData[model].reduce((sum, price) => sum + (price * energyConsumption), 0);
            const result = totalSavings - investmentCost;
            const averageMonthlySaving = totalSavings / currentForecastData[model].length;
            const paybackMonths = (averageMonthlySaving > 0) ? Math.ceil(investmentCost / averageMonthlySaving) : Infinity;
            const recommendation = result > 0 ? { text: 'Investir', class: 'invest' } : { text: 'Não Investir', class: 'no-invest' };
            
            const row = decisionTableBody.insertRow();
            row.insertCell(0).innerHTML = model;
            row.insertCell(1).innerHTML = formatCurrency(totalSavings);
            row.cells[1].title = `Média mensal: ${formatCurrency(averageMonthlySaving)}`;
            row.insertCell(2).innerHTML = formatCurrency(result);
            row.insertCell(3).innerHTML = (paybackMonths === Infinity || paybackMonths > 16) ? '> 16 meses' : `${paybackMonths} meses`;
            row.insertCell(4).innerHTML = `<span class="recommendation ${recommendation.class}">${recommendation.text}</span>`;
        });
    }

    function updateActiveButton(region) {
        regionButtons.forEach(btn => {
            btn.classList.remove('active');
            if (btn.dataset.region === region) {
                btn.classList.add('active');
            }
        });
    }

    // --- FUNÇÕES DE UTILIDADE E INTERFACE ---

    function showLoader() { document.body.classList.add('loading'); }
    function hideLoader() { document.body.classList.remove('loading'); }
    function capitalize(str) { return str.charAt(0).toUpperCase() + str.slice(1); }
    function formatCurrency(value) { return value.toLocaleString('pt-BR', { style: 'currency', currency: 'BRL' }); }

    // --- CONFIGURAÇÃO DOS EVENT LISTENERS ---

    regionButtons.forEach(button => {
        button.addEventListener('click', () => updateDashboard(button.dataset.region));
    });

    investmentCostInput.addEventListener('input', updateDecisionAnalysis);
    energyConsumptionInput.addEventListener('input', updateDecisionAnalysis);

    // Lógica do Lightbox
    const lightbox = document.getElementById('lightbox');
    const lightboxImg = document.getElementById('lightbox-img');
    const closeBtn = document.querySelector('.lightbox-close');
    
    document.querySelectorAll('.expandable-image').forEach(image => {
        image.addEventListener('click', (e) => {
            lightboxImg.src = e.currentTarget.src;
            lightbox.style.display = 'flex';
        });
    });
    
    function closeLightbox() { lightbox.style.display = 'none'; }
    closeBtn.addEventListener('click', closeLightbox);
    lightbox.addEventListener('click', e => { if (e.target === lightbox) closeLightbox(); });

    // --- CARGA INICIAL DO DASHBOARD ---
    updateDashboard('sudeste');

    async function loadAndRenderAIInsights() {
    const container = document.getElementById('ia-insights-container');
    try {
        const response = await fetch('static/data/insights_ia.json');
        if (!response.ok) throw new Error('Arquivo de insights da IA não encontrado.');
        
        const data = await response.json();
        
        container.innerHTML = ''; // Limpa o loader
        
        const title = document.createElement('h4');
        title.textContent = data.titulo;
        container.appendChild(title);
        
        const insightList = document.createElement('ul');
        data.insights.forEach(insightText => {
            const listItem = document.createElement('li');
            listItem.textContent = insightText;
            insightList.appendChild(listItem);
        });
        container.appendChild(insightList);
        
    } catch (error) {
        console.error("Erro ao carregar insights da IA:", error);
        container.innerHTML = '<p>Não foi possível carregar os insights gerados pela IA. Por favor, execute o script de geração de dados novamente.</p>';
    }
}
});