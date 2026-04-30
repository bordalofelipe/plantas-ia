// leaf.js - Lógica de classificação de folhas

let leafModel;
let selectedLeafFile = null;
let leafLabels;

function onloadLeaf() {
    console.log('Configurando interface de folhas...');
    // Event listeners para folha
    document.getElementById('removeLeaf').addEventListener('click', removeFile.bind(null, 'leaf'));

    document.getElementById('classifyLeaf').addEventListener('click', function() {
        const file = selectedLeafFile;
        if (file) {
            // Criar um elemento de imagem para processar
            const img = new Image();
            img.onload = function() {
                classifyLeafImage(img);
            };
            img.onerror = function() {
                document.getElementById('diseaseResultText').textContent = 'Erro ao carregar a imagem da folha.';
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
    loadLeafLabels();
}

async function loadLeafModel() {
    try {
        leafModel = await ort.InferenceSession.create('plant-disease.onnx');
        console.log('Modelo de folhas ONNX embarcado carregado com sucesso!');
    } catch (error) {
        console.error('Erro ao carregar modelo de folhas ONNX:', error);
        throw error;
    }
}

function preprocessLeafImageToTensorData(imageElement, width = 224, height = 224) {
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

async function loadLeafLabels() {
    try {
        const r = await fetch('diseases-categories.json');
        leafLabels = await r.json();
    }
    catch (error) {
        console.error('Erro ao carregar as labels de folhas:', error);
        alert('Erro ao carregar as labels de folhas.');
    }
}

async function classifyLeafImage(imageElement) {
    showLoading('Classificando Folha...', 'Aguarde enquanto o modelo especializado processa a imagem da folha.');
    try {
        const inputTensorData = preprocessLeafImageToTensorData(imageElement, 224, 224);
        const inputTensor = new ort.Tensor('float32', inputTensorData, [1, 3, 224, 224]);

        console.log('Classificando folha - step0');
        const outputMap = await leafModel.run({'input.1': inputTensor});//{ input: inputTensor });
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
            label: leafLabels && leafLabels[i] ? leafLabels[i] : String(i)
        }));
        console.log('step5');

        // Exibir o resultado da folha específica
        const topPrediction = predictions[0];
        if (topPrediction.score < 0.5) {
            document.getElementById('diseaseResultText').textContent = 'Nenhuma doença detectada com alta confiança. Por favor, tente outra imagem de folha.';
            document.getElementById('leafInput').style.display = 'block';
        } else {
            document.getElementById('diseaseResultText').textContent =
                `Doença: ${topPrediction.label} (Confiança: ${(topPrediction.score * 100).toFixed(2)}%)`;

            // Exibir detalhes
            document.getElementById('diseaseResultText').innerHTML += '<details><summary>Details</summary><ul>';
            for (pred of predictions) {
                document.getElementById('diseaseResultText').lastChild.innerHTML += '<li>' + pred.label + ' (' + (pred.score* 100).toFixed(2) + '%)';
            }
            document.getElementById('diseaseResultText').lastChild.innerHTML += '</ul></details>';
        }

    } catch(error) {
        console.error('Erro na classificação da folha:', error);
        document.getElementById('diseaseResultText').textContent = 'Erro na classificação da imagem da folha.';
    } finally {
        hideLoading();
    }
}
