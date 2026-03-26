// index.js - Lógica de classificação de imagem usando TensorFlow.js e MobileNet

let model;

// Carregar o modelo MobileNet quando a página carrega
async function loadModel() {
    document.getElementById('resultText').textContent = 'Carregando modelo...';
    try {
	model = await tf.loadGraphModel('model/model.json');
        //model = await mobilenet.load();
        console.log('Modelo MobileNet carregado com sucesso!');
        document.getElementById('resultText').textContent = 'Modelo carregado. Selecione uma imagem para classificar.';
    } catch (error) {
        console.error('Erro ao carregar o modelo:', error);
        document.getElementById('resultText').textContent = 'Erro ao carregar o modelo.';
    }
}

async function loadLabels() {
    try {
        const r = await fetch('model/class_idx_2_name.txt');
        labels = (await r.text()).trim().split(/\r?\n/);
    }
    catch (error) {
        console.error('Erro ao carregar as labels:', error);
        document.getElementById('resultText').textContent = 'Erro ao carregar as labels.';
    }
}

function topK(array, k=5){
    return array
        .map((v,i)=>({i,v}))
        .sort((a,b)=>b.v-a.v)
        .slice(0,k);
}

// Função para classificar a imagem
async function classifyImage(imageElement) {
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
        const outputs = await model.executeAsync({'input:0': t}, ['Identity:0']);
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
            label: labels && labels[i] ? labels[i] : String(i)
        }));
        console.log('stepC');
        tf.dispose([t, outputs, logits, probs]);

        // Exibir o resultado principal
        const topPrediction = predictions[0];
        document.getElementById('resultText').textContent = 
            `Classe: ${topPrediction.label} (Confiança: ${(topPrediction.score * 100).toFixed(2)}%)`;
	// Exibir demais dentro de details
        document.getElementById('resultText').innerHTML += '<details><summary>Details</summary><ul>';
        for (pred of predictions) {
            document.getElementById('resultText').lastChild.innerHTML += '<li>' + pred.label + ' (' + (pred.score* 100).toFixed(2) + '%)';
	}
    } catch(error) {
        console.error('Erro na classificação:', error);
        document.getElementById('resultText').textContent = 'Erro na classificação da imagem.';
    }
}

/*
async function ClassifyImage(imageElement) {
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
*/

// Event listener para o input de arquivo
onload = function() {
    // Registrar Service Worker
    if ('serviceWorker' in navigator) {
        navigator.serviceWorker.register('sw.js').then((registration) => {
            console.log('Service Worker registrado com sucesso:', registration);
        }).catch((error) => {
            console.log('Erro ao registrar Service Worker:', error);
        });
    }

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
    loadLabels();
}
