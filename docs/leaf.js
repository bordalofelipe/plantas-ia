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
}

async function loadLeafModel() {
    try {
        leafModel = await ort.InferenceSession.create('leaf-modelo-embedded.onnx');
        console.log('Modelo de folhas ONNX embarcado carregado com sucesso!');
    } catch (error) {
        console.error('Erro ao carregar modelo de folhas ONNX:', error);
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

async function loadLeafLabels() {
    try {
        const r = await fetch('class_idx_2_name.txt');
        leafLabels = (await r.text()).trim().split(/\r?\n/);
    }
    catch (error) {
        console.error('Erro ao carregar as labels de folhas:', error);
        alert('Erro ao carregar as labels de folhas.');
    }
}

async function classifyLeafImage(imageElement) {
    showLoading('Classificando Folha...', 'Aguarde enquanto o modelo especializado processa a imagem da folha.');
    try {
        const inputTensorData = preprocessImageToTensorData(imageElement, 224, 224);
        const inputTensor = new ort.Tensor('float32', inputTensorData, [1, 3, 224, 224]);

        console.log('Classificando folha - step0');
        const outputMap = await leafModel.run({ input: inputTensor });
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
        document.getElementById('diseaseResultText').textContent =
            `Doença: ${topPrediction.label} (Confiança: ${(topPrediction.score * 100).toFixed(2)}%)`;

        // Exibir detalhes
        document.getElementById('diseaseResultText').innerHTML += '<details><summary>Details</summary><ul>';
        for (pred of predictions) {
            document.getElementById('diseaseResultText').lastChild.innerHTML += '<li>' + pred.label + ' (' + (pred.score* 100).toFixed(2) + '%)';
        }
        document.getElementById('diseaseResultText').lastChild.innerHTML += '</ul></details>';

        if (topPrediction.score < 0.5) {
            document.getElementById('leafInput').style.display = 'block';
        }

    } catch(error) {
        console.error('Erro na classificação da folha:', error);
        document.getElementById('diseaseResultText').textContent = 'Erro na classificação da imagem da folha.';
    } finally {
        hideLoading();
    }
}