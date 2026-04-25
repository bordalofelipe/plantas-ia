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
        plantModel = await ort.InferenceSession.create('plant-modelo-embedded.onnx');
        console.log('Modelo de plantas ONNX embarcado carregado com sucesso!');
    } catch (error) {
        console.error('Erro ao carregar o modelo de plantas ONNX:', error);
        throw error;
    }
}

function preprocessImageToTensorData(imageElement, width = 224, height = 224) {
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(imageElement, 0, 0, width, height);
    const imageData = ctx.getImageData(0, 0, width, height).data;
    const data = new Float32Array(3 * width * height);
    const hw = width * height;

    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const px = (y * width + x) * 4;
            const r = imageData[px] / 255.0;
            const g = imageData[px + 1] / 255.0;
            const b = imageData[px + 2] / 255.0;
            const idx = y * width + x;
            data[idx] = r;
            data[hw + idx] = g;
            data[2 * hw + idx] = b;
        }
    }

    return data;
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
        const inputTensorData = preprocessImageToTensorData(imageElement, 224, 224);
        const inputTensor = new ort.Tensor('float32', inputTensorData, [1, 224, 224, 3]);

        console.log('step0');
        const outputMap = await plantModel.run( {'args_0:0': inputTensor}); // { input: inputTensor });
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
