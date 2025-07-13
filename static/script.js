document.addEventListener('DOMContentLoaded', function() {

    // --- ELEMENTOS GLOBAIS ---
    const regionButtons = document.querySelectorAll('.region-btn');
    const investmentCostInput = document.getElementById('investmentCost');
    const energyConsumptionInput = document.getElementById('energyConsumption');
    let currentForecastData = {}; // Armazena os dados da região atual

    // --- FUNÇÕES PRINCIPAIS ---

    function updateDashboard(region) {
        // Atualiza os títulos
        document.getElementById('outliers-title').textContent = `Passo 2: Preparação dos Dados (Região ${capitalize(region)})`;
        document.getElementById('decomposition-card-title').textContent = `Decomposição da Série Temporal - ${capitalize(region)}`;
        
        // Atualiza as imagens dos gráficos
        const cacheBuster = `?t=${new Date().getTime()}`;
        document.getElementById('decompositionChart').src = `static/images/decomposition_${region}.png${cacheBuster}`;
        document.getElementById('outliersChart').src = `static/images/outliers_${region}.png${cacheBuster}`;
        document.getElementById('forecastChart').src = `static/images/forecast_${region}.png${cacheBuster}`;

        // Carrega os dados da região e, quando terminar, atualiza a tabela e a análise de decisão
        loadForecastData(region).then(() => {
            renderForecastTable();
            updateDecisionAnalysis();
        });

        // Atualiza a classe 'active' nos botões
        regionButtons.forEach(btn => {
            btn.classList.remove('active');
            if (btn.dataset.region === region) {
                btn.classList.add('active');
            }
        });
    }

    async function loadForecastData(region) {
        const dataFile = `static/data/forecast_data_${region}.json`;
        try {
            const response = await fetch(dataFile);
            if (!response.ok) throw new Error(`Arquivo de dados não encontrado para a região ${region}`);
            currentForecastData = await response.json();
        } catch (error) {
            console.error('Erro ao carregar dados de previsão:', error);
            currentForecastData = {}; // Limpa dados em caso de erro
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
            // Formata os números para exibição como moeda
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
            // Calcula a economia total
            const totalSavings = currentForecastData[model].reduce((sum, price) => sum + (price * energyConsumption), 0);
            
            // Calcula o resultado
            const result = totalSavings - investmentCost;

            // Calcula o payback
            const averageMonthlySaving = totalSavings / currentForecastData[model].length;
            const paybackMonths = (averageMonthlySaving > 0) ? Math.ceil(investmentCost / averageMonthlySaving) : Infinity;

            // Define a recomendação
            const recommendation = result > 0 ? { text: 'Investir', class: 'invest' } : { text: 'Não Investir', class: 'no-invest' };
            
            // Cria a linha da tabela
            const row = decisionTableBody.insertRow();
            row.insertCell(0).innerHTML = model;
            row.insertCell(1).innerHTML = formatCurrency(totalSavings);
            row.cells[1].title = `Média mensal: ${formatCurrency(averageMonthlySaving)}`;
            row.insertCell(2).innerHTML = formatCurrency(result);
            row.insertCell(3).innerHTML = (paybackMonths === Infinity || paybackMonths > 16) ? '> 16 meses' : `${paybackMonths} meses`;
            row.insertCell(4).innerHTML = `<span class="recommendation ${recommendation.class}">${recommendation.text}</span>`;
        });
    }

    // --- EVENT LISTENERS ---

    regionButtons.forEach(button => {
        button.addEventListener('click', () => updateDashboard(button.dataset.region));
    });

    // Atualiza a análise de decisão ao mudar os valores nos inputs
    investmentCostInput.addEventListener('input', updateDecisionAnalysis);
    energyConsumptionInput.addEventListener('input', updateDecisionAnalysis);

    // --- FUNÇÕES UTILITÁRIAS ---

    function capitalize(str) { return str.charAt(0).toUpperCase() + str.slice(1); }
    function formatCurrency(value) { return value.toLocaleString('pt-BR', { style: 'currency', currency: 'BRL' }); }

    // --- LÓGICA DO LIGHTBOX (permanece a mesma) ---
    const lightbox = document.getElementById('lightbox');
    const lightboxImg = document.getElementById('lightbox-img');
    const closeBtn = document.querySelector('.lightbox-close');
    document.querySelectorAll('.expandable-image').forEach(image => {
        image.addEventListener('click', (e) => {
            // Atualiza o src da imagem no lightbox antes de mostrá-la
            lightboxImg.src = e.currentTarget.src;
            lightbox.style.display = 'flex';
        });
    });
    function closeLightbox() { lightbox.style.display = 'none'; }
    closeBtn.addEventListener('click', closeLightbox);
    lightbox.addEventListener('click', e => { if (e.target === lightbox) closeLightbox(); });

    // --- CARGA INICIAL ---
    updateDashboard('sudeste');
});