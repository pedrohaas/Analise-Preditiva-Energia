// Executa o código quando o conteúdo da página estiver totalmente carregado
document.addEventListener('DOMContentLoaded', function() {
    const resultsSection = document.getElementById('results');

    // Caminho para os arquivos de imagem e dados gerados
    const outliersChartSrc = 'static/images/outliers_sudeste.png';
    const forecastChartSrc = 'static/images/forecast_sudeste.png';
    const dataFile = 'static/data/forecast_data.json';

    // Faz a chamada para o arquivo de dados JSON local
    fetch(dataFile)
        .then(response => {
            if (!response.ok) {
                throw new Error('Não foi possível carregar o arquivo de dados: ' + response.statusText);
            }
            return response.json();
        })
        .then(data => {
            // Atualiza as imagens dos gráficos
            document.getElementById('outliersChart').src = outliersChartSrc;
            document.getElementById('forecastChart').src = forecastChartSrc;
            
            // Preenche a tabela de previsões
            const tableBody = document.getElementById('forecastTable').getElementsByTagName('tbody')[0];
            tableBody.innerHTML = ''; // Limpa a tabela
            
            for (let i = 0; i < data.Data.length; i++) {
                let row = tableBody.insertRow();
                row.insertCell(0).innerHTML = data.Data[i];
                row.insertCell(1).innerHTML = data.Prophet[i];
                row.insertCell(2).innerHTML = data.ARIMA[i];
                row.insertCell(3).innerHTML = data.SARIMA[i];
            }

            // Mostra os resultados
            resultsSection.style.display = 'block';
        })
        .catch(error => {
            console.error('Erro ao carregar dados estáticos:', error);
            alert('Ocorreu um erro ao carregar os resultados da análise. Verifique o console para mais detalhes.');
        });
});