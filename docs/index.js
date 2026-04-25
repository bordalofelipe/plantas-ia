// index.js - Utilitários compartilhados

function showLoading(title = 'Aguarde...', text = 'Processando...') {
    document.getElementById('loadingTitle').textContent = title;
    document.getElementById('loadingText').textContent = text;
    document.getElementById('loading').style.display = 'block';
}

function hideLoading() {
    document.getElementById('loading').style.display = 'none';
}

function prepareDropzone(type) {
    console.log(`Preparando dropzone para ${type}...`);
    const dropzone = document.getElementById(`dropzone${type === 'plant' ? 'Plant' : 'Leaf'}`);
    const fileInput = document.getElementById(`imageInput${type === 'plant' ? 'Plant' : 'Leaf'}`);
    const preview = document.getElementById(`preview${type === 'plant' ? 'Plant' : 'Leaf'}`);
    
    dropzone.addEventListener('click', () => fileInput.click());

    dropzone.addEventListener('dragover', (event) => {
        event.preventDefault();
        dropzone.classList.add('dragover');
    });

    dropzone.addEventListener('dragleave', () => {
        dropzone.classList.remove('dragover');
    });

    dropzone.addEventListener('drop', (event) => {
        event.preventDefault();
        dropzone.classList.remove('dragover');
        const file = event.dataTransfer.files[0];
        handleFileUpload(file, type);
    });

    fileInput.addEventListener('change', (event) => {
        const file = event.target.files[0];
        handleFileUpload(file, type);
    });
}

function removeFile(type) {
    if (type === 'plant') {
        selectedPlantFile = null;
    } else {
        selectedLeafFile = null;
    }
    document.getElementById(`imageInput${type === 'plant' ? 'Plant' : 'Leaf'}`).value = '';
    document.getElementById(`dropzone${type === 'plant' ? 'Plant' : 'Leaf'}`).style.display = '';
    document.getElementById(`remove${type === 'plant' ? 'Plant' : 'Leaf'}`).style.display = 'none';
    document.getElementById(`preview${type === 'plant' ? 'Plant' : 'Leaf'}`).style.display = 'none';
}

function setSelectedFile(file, type) {
    if (type === 'plant') {
        selectedPlantFile = file;
    } else {
        selectedLeafFile = file;
    }
    console.log(`Arquivo ${type === 'plant' ? 'da planta' : 'da folha'} selecionado:`, file);
    console.log(`Preview${type === 'plant' ? 'planta' : 'folha'}:`, document.getElementById(`preview${type === 'plant' ? 'Plant' : 'Leaf'}`));
    document.getElementById(`preview${type === 'plant' ? 'Plant' : 'Leaf'}`).src = URL.createObjectURL(file);
    document.getElementById(`dropzone${type === 'plant' ? 'Plant' : 'Leaf'}`).style.display = 'none';
    document.getElementById(`remove${type === 'plant' ? 'Plant' : 'Leaf'}`).style.display = '';
    document.getElementById(`preview${type === 'plant' ? 'Plant' : 'Leaf'}`).style.display = '';
}

function handleFileUpload(file, type) {
    removeFile(type);
    if (!file) return;
    if (!file.type.startsWith('image/')) {
        alert('Apenas imagens são aceitas.');
        return;
    }
    setSelectedFile(file, type);
}

function topK(array, k=5){
    return array
        .map((v,i)=>({i,v}))
        .sort((a,b)=>b.v-a.v)
        .slice(0,k);
}

// Event listener para o input de arquivo
onload = async function() {
    // Registrar Service Worker
    if ('serviceWorker' in navigator) {
        navigator.serviceWorker.register('sw.js').then((registration) => {
            console.log('Service Worker registrado com sucesso:', registration);
        }).catch((error) => {
            console.log('Erro ao registrar Service Worker:', error);
        });
    }

    // Carregar ambos os modelos sequencialmente
    showLoading('Carregando Modelos...', 'Carregando modelos especializados para plantas e folhas...');
    try {
        await loadPlantModel();
        await loadLeafModel();
        console.log('Ambos os modelos carregados com sucesso!');
    } catch (error) {
        console.error('Erro ao carregar modelos:', error);
        alert('Erro ao carregar os modelos.');
    } finally {
        hideLoading();
    }

    // Configurar interfaces
    prepareDropzone('leaf');
    prepareDropzone('plant');
    onloadPlant();
    onloadLeaf();
}