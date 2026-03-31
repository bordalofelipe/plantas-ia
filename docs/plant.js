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
            img.onload = function() {
                classifyPlantImage(img);
                classifyLeafImage(img); // Classificar também como folha para comparação
            };
            img.onerror = function() {
                document.getElementById('resultText').textContent = 'Erro ao carregar a imagem.';
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
	plantModel = await tf.loadGraphModel('model/model.json');
        // Alternativa: plantModel = await tf.loadGraphModel('model/plant_model.json');
        console.log('Modelo de plantas carregado com sucesso!');
    } catch (error) {
        console.error('Erro ao carregar o modelo de plantas:', error);
        throw error; // Re-throw para que o chamador possa lidar
    }
}

async function loadPlantLabels() {
    try {
        const r = await fetch('model/class_idx_2_name.txt');
        plantLabels = (await r.text()).trim().split(/\r?\n/);
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
	// Preprocess: fromPixels -> resize -> normalize -> transpose to channels-first -> batch
        console.log('step0');
        let t = tf.browser.fromPixels(imageElement).toFloat(); // [H,W,3]
        console.log('step1');
        t = tf.image.resizeBilinear(t, [224,224]);     // [224,224,3]
        // Ajuste normalização conforme seu treino:
        // Usar [0,1]:
        console.log('step2');
        t = t.div(255.0);
        // Se treinou com [-1,1], substitua por:
        // t = t.div(127.5).sub(1);

        // transpose to [3,224,224]
        console.log('step3');
        t = tf.transpose(t, [2,0,1]);
        console.log('step4');
        t = t.expandDims(0); // [1,3,224,224]

        // Execute usando nomes do model.json
        console.log('step5');
        const outputs = await plantModel.executeAsync({'input:0': t}, ['Identity:0']);
        console.log('step6');
        const logits = Array.isArray(outputs) ? outputs[0] : outputs;
        console.log('step7');
        const probs = tf.softmax(logits);
        console.log('step8');
        const arr = await probs.data(); // flat Float32Array length 1081
        console.log('step9');
        // convert to JS array
        const jsArr = Array.from(arr);
        console.log('stepA');
        // get top-5
        const top = topK(jsArr, 5);
        console.log('stepB');
        // map to labels (guard)
        const predictions = top.map(({i,v}) => ({
            index: i,
            score: v,
            label: plantLabels && plantLabels[i] ? plantLabels[i] : String(i)
        }));
        console.log('stepC');
        tf.dispose([t, outputs, logits, probs]);

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