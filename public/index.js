// index.js - Lógica de classificação de imagem usando TensorFlow.js e MobileNet

let model;

// Carregar o modelo MobileNet quando a página carrega
async function loadModel() {
    document.getElementById('resultText').textContent = 'Carregando modelo...';
    try {
        model = await mobilenet.load();
        console.log('Modelo MobileNet carregado com sucesso!');
        document.getElementById('resultText').textContent = 'Modelo carregado. Selecione uma imagem para classificar.';
    } catch (error) {
        console.error('Erro ao carregar o modelo:', error);
        document.getElementById('resultText').textContent = 'Erro ao carregar o modelo.';
    }
}

// Função para classificar a imagem
async function classifyImage(imageElement) {
    try {
        const predictions = await model.classify(imageElement);
        console.log('Predições:', predictions);
        
        // Exibir o resultado principal
        const topPrediction = predictions[0];
        document.getElementById('resultText').textContent = 
            `Classe: ${topPrediction.className} (Confiança: ${(topPrediction.probability * 100).toFixed(2)}%)`;
	// Exibir demais dentro de details
        document.getElementById('resultText').innerHTML += '<details><summary>Details</summary><ul>';
        for (pred of predictions) {
            document.getElementById('resultText').lastChild.innerHTML += '<li>' + pred.className + ' (' + (pred.probability * 100).toFixed(2) + '%)';
	}
    } catch (error) {
        console.error('Erro na classificação:', error);
        document.getElementById('resultText').textContent = 'Erro na classificação da imagem.';
    }
}

// Event listener para o input de arquivo
onload = function() {
    document.getElementById('classify').addEventListener('click', function() {
        const file = document.getElementById('imageInput').files[0];
        if (file) {
            document.getElementById('resultText').textContent = 'Classificando...';
            
            // Criar um elemento de imagem para processar
            const img = new Image();
            img.onload = function() {
                classifyImage(img);
            };
            img.onerror = function() {
                document.getElementById('resultText').textContent = 'Erro ao carregar a imagem.';
            };
            
            // Ler o arquivo como URL de dados
            const reader = new FileReader();
            reader.onload = function(e) {
                img.src = e.target.result;
            };
            reader.readAsDataURL(file);
        }
    });
    loadModel();
}
