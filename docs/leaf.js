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
                document.getElementById('resultText').textContent = 'Erro ao carregar a imagem da folha.';
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
        leafModel = await tf.loadGraphModel('model/model.json');
        // Alternativa: leafModel = await tf.loadGraphModel('model/leaf_model.json');
        console.log('Modelo de folhas carregado com sucesso!');
    } catch (error) {
        console.error('Erro ao carregar modelo de folhas:', error);
        throw error; // Re-throw para que o chamador possa lidar
    }
}

async function loadLeafLabels() {
    try {
        const r = await fetch('model/class_idx_2_name.txt');
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
	// Preprocess: fromPixels -> resize -> normalize -> transpose to channels-first -> batch
        console.log('Classificando folha - step0');
        let t = tf.browser.fromPixels(imageElement).toFloat(); // [H,W,3]
        console.log('step1');
        t = tf.image.resizeBilinear(t, [224,224]);     // [224,224,3]
        // Ajuste normalização conforme seu treino:
        // Usar [0,1]:
        console.log('step2');
        t = t.div(255.0);

        // transpose to [3,224,224]
        console.log('step3');
        t = tf.transpose(t, [2,0,1]);
        console.log('step4');
        t = t.expandDims(0); // [1,3,224,224]

        // Execute usando nomes do model.json
        console.log('step5');
        const outputs = await leafModel.executeAsync({'input:0': t}, ['Identity:0']);
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
            label: leafLabels && leafLabels[i] ? leafLabels[i] : String(i)
        }));
        console.log('stepC');
        tf.dispose([t, outputs, logits, probs]);

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
        document.getElementById('resultText').textContent = 'Erro na classificação da imagem da folha.';
    } finally {
        hideLoading();
    }
}