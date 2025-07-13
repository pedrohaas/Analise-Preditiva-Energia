document.addEventListener('DOMContentLoaded', function() {

    const regionButtons = document.querySelectorAll('.region-btn');

    // Função principal para atualizar todo o conteúdo da página
    function updateDashboard(region) {
        // Atualiza os títulos
        document.getElementById('outliers-title').textContent = `Passo 1: Preparação dos Dados (Região ${capitalize(region)})`;
        // Adicione aqui a atualização de outros títulos se desejar...

        // Atualiza as imagens dos gráficos
        const cacheBuster = `?t=${new Date().getTime()}`; // Evita problemas de cache
        document.getElementById('outliersChart').src = `static/images/outliers_${region}.png${cacheBuster}`;
        document.getElementById('forecastChart').src = `static/images/forecast_${region}.png${cacheBuster}`;

        // Carrega e renderiza a nova tabela de dados
        loadTableData(region);

        // Atualiza a classe 'active' nos botões
        regionButtons.forEach(btn => {
            btn.classList.remove('active');
            if (btn.dataset.region === region) {
                btn.classList.add('active');
            }
        });
    }

    // Função para carregar os dados da tabela de um arquivo JSON
    function loadTableData(region) {
        const dataFile = `static/data/forecast_data_${region}.json`;
        fetch(dataFile)
            .then(response => {
                if (!response.ok) throw new Error(`Arquivo de dados não encontrado para a região ${region}`);
                return response.json();
            })
            .then(data => {
                const table = document.getElementById('forecastTable');
                table.innerHTML = ''; // Limpa a tabela antiga

                let thead = table.createTHead();
                let row = thead.insertRow();
                for (let key in data) {
                    let th = document.createElement("th");
                    th.innerHTML = key;
                    row.appendChild(th);
                }
                
                let tbody = table.createTBody();
                for (let i = 0; i < data.Data.length; i++) {
                    let row = tbody.insertRow();
                    row.insertCell(0).innerHTML = data.Data[i];
                    row.insertCell(1).innerHTML = data.Prophet[i];
                    row.insertCell(2).innerHTML = data.ARIMA[i];
                    row.insertCell(3).innerHTML = data.SARIMA[i];
                }
            })
            .catch(error => console.error('Erro ao carregar tabela:', error));
    }

    // Adiciona o evento de clique para cada botão de região
    regionButtons.forEach(button => {
        button.addEventListener('click', () => {
            const region = button.dataset.region;
            updateDashboard(region);
        });
    });

    // Função utilitária para capitalizar a primeira letra
    function capitalize(str) {
        return str.charAt(0).toUpperCase() + str.slice(1);
    }

    // --- LÓGICA DO LIGHTBOX (permanece a mesma) ---
    const lightbox = document.getElementById('lightbox');
    const lightboxImg = document.getElementById('lightbox-img');
    const closeBtn = document.querySelector('.lightbox-close');
    const images = document.querySelectorAll('.expandable-image');
    images.forEach(image => image.addEventListener('click', () => {
        lightbox.style.display = 'flex';
        lightboxImg.src = image.src;
    }));
    function closeLightbox() { lightbox.style.display = 'none'; }
    closeBtn.addEventListener('click', closeLightbox);
    lightbox.addEventListener('click', e => { if (e.target === lightbox) closeLightbox(); });

    // Carga inicial do dashboard com a região padrão (Sudeste)
    updateDashboard('sudeste');
});