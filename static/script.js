document.addEventListener('DOMContentLoaded', function() {
    
    // --- LÓGICA EXISTENTE PARA CARREGAR DADOS ---
    const dataFile = 'static/data/forecast_data.json';

    fetch(dataFile)
        .then(response => {
            if (!response.ok) {
                throw new Error('Não foi possível carregar o arquivo de dados: ' + response.statusText);
            }
            return response.json();
        })
        .then(data => {
            const table = document.getElementById('forecastTable');
            // Cria o cabeçalho da tabela
            let thead = table.createTHead();
            let row = thead.insertRow();
            for (let key in data) {
                let th = document.createElement("th");
                th.innerHTML = key;
                row.appendChild(th);
            }
            // Cria o corpo da tabela
            let tbody = table.createTBody();
            for (let i = 0; i < data.Data.length; i++) {
                let row = tbody.insertRow();
                row.insertCell(0).innerHTML = data.Data[i];
                row.insertCell(1).innerHTML = data.Prophet[i];
                row.insertCell(2).innerHTML = data.ARIMA[i];
                row.insertCell(3).innerHTML = data.SARIMA[i];
            }
        })
        .catch(error => {
            console.error('Erro ao carregar dados estáticos:', error);
            alert('Ocorreu um erro ao carregar os resultados da análise.');
        });

    // --- NOVA LÓGICA PARA O LIGHTBOX ---
    const lightbox = document.getElementById('lightbox');
    const lightboxImg = document.getElementById('lightbox-img');
    const closeBtn = document.querySelector('.lightbox-close');

    // Pega todas as imagens que podem ser expandidas
    const images = document.querySelectorAll('.expandable-image');

    // Adiciona o evento de clique para cada imagem
    images.forEach(image => {
        image.addEventListener('click', () => {
            lightbox.style.display = 'flex'; // Mostra o overlay
            lightboxImg.src = image.src;      // Define a imagem a ser mostrada
        });
    });

    // Função para fechar o lightbox
    function closeLightbox() {
        lightbox.style.display = 'none';
    }

    // Fecha ao clicar no botão 'x'
    closeBtn.addEventListener('click', closeLightbox);

    // Fecha também ao clicar fora da imagem (no overlay)
    lightbox.addEventListener('click', (e) => {
        if (e.target === lightbox) {
            closeLightbox();
        }
    });
});