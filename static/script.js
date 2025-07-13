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

        // Carrega os dados de previsão e validação em paralelo para mais velocidade
        await Promise.all([
            loadForecastData(region),
            loadValidationData()
        ]);

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
        const regionMetrics = allValidationMetrics[region];
        if (!regionMetrics) return;

        const summaryEl = document.getElementById('validation-summary');
        summaryEl.innerHTML = `Para a região <strong>${capitalize(region)}</strong>, foram detectados e tratados <strong>${regionMetrics.outliers_count} outliers</strong> (${regionMetrics.outliers_percentage}% do total de dados).`;

        const tableBody = document.querySelector("#validation-table tbody");
        tableBody.innerHTML = '';
        
        for (const modelName in regionMetrics.metrics) {
            const metrics = regionMetrics.metrics[modelName];
            const row = tableBody.insertRow();
            row.insertCell(0).textContent = modelName;
            
            // --- CORREÇÃO AQUI: Verifica se o valor é nulo antes de formatar ---
            row.insertCell(1).textContent = metrics.MAE !== null ? metrics.MAE.toFixed(2) : 'N/A';
            row.insertCell(2).textContent = metrics.MSE !== null ? metrics.MSE.toFixed(2) : 'N/A';
            row.insertCell(3).textContent = metrics.RMSE !== null ? metrics.RMSE.toFixed(2) : 'N/A';
        }
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
});