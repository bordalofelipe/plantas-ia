// plant.js - Lógica de classificação de plantas

let plantModel;
let selectedPlantFile = null;
let plantLabels;

function onloadPlant() {
    console.log('Configurando interface de plantas...'); 
    // Event listeners para planta
    document.getElementById('removePlant').addEventListener('click', removeFile.bind(null, 'plant'));

    document.getElementById('classifyPlant').addEventListener('click', function() {
        const file = selectedPlantFile;
        if (file) {
            // Criar um elemento de imagem para processar
            const img = new Image();
            img.onload = async function() {
                await classifyPlantImage(img);
                await classifyLeafImage(img); // Classificar também como folha para comparação
            };
            img.onerror = function() {
                document.getElementById('speciesResultText').textContent = 'Erro ao carregar a imagem.';
                hideLoading();
            };
            
            // Ler o arquivo como URL de dados
            const reader = new FileReader();
            reader.onload = function(e) {
                img.src = e.target.result;
            };
            reader.readAsDataURL(file);
        }
    });
    loadPlantLabels();
}

async function loadPlantModel() {
    try {
        plantModel = await ort.InferenceSession.create('plant-species.onnx');
        console.log('Modelo de plantas ONNX embarcado carregado com sucesso!');
    } catch (error) {
        console.error('Erro ao carregar o modelo de plantas ONNX:', error);
        throw error;
    }
}

function preprocessPlantImageToTensorData(imageElement, width = 224, height = 224) {
    // Step 1: Resize to 232x232 (resize_size)
    const resizeCanvas = document.createElement('canvas');
    resizeCanvas.width = 232;
    resizeCanvas.height = 232;
    const resizeCtx = resizeCanvas.getContext('2d');
    resizeCtx.drawImage(imageElement, 0, 0, 232, 232);

    // Step 2: Center crop to 224x224 (crop_size)
    const cropCanvas = document.createElement('canvas');
    cropCanvas.width = 224;
    cropCanvas.height = 224;
    const cropCtx = cropCanvas.getContext('2d');
    // Center crop: offset by (232-224)/2 = 4 pixels
    cropCtx.drawImage(resizeCanvas, 4, 4, 224, 224, 0, 0, 224, 224);

    const data = cropCtx.getImageData(0, 0, 224, 224).data;
    const tensorData = new Float32Array(3 * 224 * 224); // Canais x Altura x Largura
    for (let i = 0; i < data.length; i += 4) {
        // Convert to RGB and normalize to [0, 1]
        const r = data[i] / 255;
        const g = data[i + 1] / 255;
        const b = data[i + 2] / 255;
        
        tensorData[i / 4] = r; // Canal R
        tensorData[(i / 4) + (224 * 224)] = g; // Canal G
        tensorData[(i / 4) + (2 * 224 * 224)] = b; // Canal B
    }
    return tensorData;
}

function softmax(array) {
    const max = Math.max(...array);
    const exps = array.map((value) => Math.exp(value - max));
    const sum = exps.reduce((acc, value) => acc + value, 0);
    return exps.map((value) => value / sum);
}

async function loadPlantLabels() {
    try {
        const r = await fetch('species-categories.json');
        plantLabels = await r.json();
    }
    catch (error) {
        console.error('Erro ao carregar as labels de plantas:', error);
        alert('Erro ao carregar as labels de plantas.');
    }
}

async function classifyPlantImage(imageElement) {
    document.getElementById('leafInput').style.display = 'none'; // Esconder input de folha inicialmente
    showLoading('Classificando...', 'Aguarde enquanto o modelo processa a imagem da planta e folha.');
    try {
        const inputTensorData = preprocessPlantImageToTensorData(imageElement, 224, 224);
        console.log('Input tensor criado:', inputTensorData);
        const inputTensor = new ort.Tensor('float32', inputTensorData, [1, 3, 224, 224]);
        console.log('step0');
        const outputMap = await plantModel.run({'input.1': inputTensor});
        console.log('step1');
        const outputTensor = outputMap.output || outputMap[Object.keys(outputMap)[0]];
        const jsArr = Array.from(outputTensor.data);
        console.log('step2');
        const probsArr = softmax(jsArr);
        console.log('step3');
        const top = topK(probsArr, 5);
        console.log('step4');
        const predictions = top.map(({i,v}) => ({
            index: i,
            score: v,
            label: plantLabels && plantLabels[i] ? plantLabels[i] : String(i)
        }));
        console.log('step5');

        // Resultados da classificação
        const plantPrediction = predictions[0];

        // Exibir resultados
        let resultText = `Planta - Classe: ${plantPrediction.label} (Confiança: ${(plantPrediction.score * 100).toFixed(2)}%)\n`;

        document.getElementById('speciesResultText').textContent = resultText;

        // Exibir detalhes
        document.getElementById('speciesResultText').innerHTML += '<details><summary>Details</summary><ul>';
        for (pred of predictions) {
            document.getElementById('speciesResultText').lastChild.innerHTML += '<li>' + pred.label + ' (' + (pred.score* 100).toFixed(2) + '%)';
        }
        document.getElementById('speciesResultText').lastChild.innerHTML += '</ul></details>';

    } catch(error) {
        console.error('Erro na classificação:', error);
        document.getElementById('speciesResultText').textContent = 'Erro na classificação da imagem.';
    } finally {
        hideLoading();
    }
}
