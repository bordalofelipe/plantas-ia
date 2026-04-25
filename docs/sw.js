// Service Worker para Plantas IA
const CACHE_NAME = 'plantas-ia-v1';
const urlsToCache = [
  '/plantas-ia/',
  '/plantas-ia/index.html',
  '/plantas-ia/index.js',
  '/plantas-ia/plant.js',
  '/plantas-ia/leaf.js',
  '/plantas-ia/manifest.webmanifest',
  '/plantas-ia/leaf-modelo-embedded.onnx',
  '/plantas-ia/plant-modelo-embedded.onnx',
  '/plantas-ia/class_idx_2_name.txt',
  '/plantas-ia/styles.css',
];

// Instalação do Service Worker
self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
      console.log('Cache aberto');
      return cache.addAll(urlsToCache).catch((error) => {
        console.warn('Erro ao fazer cache de alguns arquivos:', error);
        // Continua mesmo se alguns arquivos falharem
        return cache.addAll(urlsToCache.filter((url) => url !== '/manifest.webmanifest'));
      });
    })
  );
  self.skipWaiting();
});

// Ativação do Service Worker
self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((cacheNames) => {
      return Promise.all(
        cacheNames.map((cacheName) => {
          if (cacheName !== CACHE_NAME) {
            console.log('Deletando cache antigo:', cacheName);
            return caches.delete(cacheName);
          }
        })
      );
    })
  );
  self.clients.claim();
});

// Estratégia de cache: Cache first para tudo
self.addEventListener('fetch', (event) => {
  // Ignorar requisições não-GET
  if (event.request.method !== 'GET') {
    return;
  }

  // Cache first para todos os recursos
  event.respondWith(
    caches.match(event.request).then((cachedResponse) => {
      if (cachedResponse) {
        // Atualizar cache em background quando tiver conexão
        fetch(event.request)
          .then((response) => {
            if (response && response.status === 200) {
              const responseToCache = response.clone();
              caches.open(CACHE_NAME).then((cache) => {
                cache.put(event.request, responseToCache);
              });
            }
          })
          .catch(() => {
            // Silenciosamente falha em background
          });
        return cachedResponse;
      }

      return fetch(event.request)
        .then((response) => {
          // Cache de respostas bem-sucedidas
          if (!response || response.status !== 200 || response.type === 'error') {
            return response;
          }

          const responseToCache = response.clone();
          caches.open(CACHE_NAME).then((cache) => {
            cache.put(event.request, responseToCache);
          });

          return response;
        })
        .catch(() => {
          // Se falhar e não tem cache, retorna página offline
          if (event.request.destination === 'document') {
            return caches.match('/index.html');
          }
        });
    })
  );
});
