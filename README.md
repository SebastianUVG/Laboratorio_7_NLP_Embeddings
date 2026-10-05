# Laboratorio 7 — NLP end-to-end y embeddings

Implementación reproducible de WikiText-103, SGNS propio en PyTorch, Word2Vec skip-gram de gensim, GloVe 100d y clasificación AG News.

## Instalación y ejecución

### Kaggle (recomendado para el entrenamiento completo)

Suba y ejecute `notebooks/laboratorio_7_nlp_embeddings_kaggle.ipynb`. En la configuración del notebook active **GPU** e **Internet**. Esta versión es autocontenida, guarda checkpoints reanudables en `/kaggle/working/lab7` y al final genera `lab7_resultados.zip`.

### Ejecución local

```bash
python -m venv .venv
.venv/Scripts/activate       # Windows
pip install -r requirements.txt
jupyter notebook notebooks/laboratorio_7_nlp_embeddings.ipynb
```

El notebook descarga los datos sólo al ejecutarse. Los datasets y checkpoints no se versionan. El subconjunto principal se fija en 20 millones de tokens y se reutiliza para SGNS y gensim; los experimentos secundarios se declaran en `src/config.py`.

## Estructura

`src/` contiene el código reusable; `artifacts/` contiene figuras, tablas, embeddings y checkpoints; `results/` contiene CSV incrementales.

Cada epoch de SGNS guarda un checkpoint con modelo, optimizador, configuración, pérdida y métricas. Si el checkpoint compatible existe, el entrenamiento continúa desde el siguiente epoch. Los resultados no ejecutados permanecen explícitamente como no ejecutados; no se inventan métricas.

## Dependencias y datos

Las fuentes exigidas son `Salesforce/wikitext` (`wikitext-103-raw-v1`), `fancyzhx/ag_news` y `glove-wiki-gigaword-100`. `questions-words.txt`, `wordsim353.tsv` y `simlex999.txt` se obtienen desde `gensim.test.utils.datapath`.

## CPU, CUDA y memoria

El dispositivo se selecciona automáticamente: CUDA si PyTorch la detecta y CPU en caso contrario. Para forzar una opción:

```powershell
$env:LAB7_DEVICE="cuda"   # o "cpu"
```

Los pares skip-gram se generan lazy en CPU y no se materializan completos en RAM; sólo los batches se transfieren a GPU. Se activa mixed precision en CUDA y se liberan cachés entre experimentos. Si se agota la memoria de GPU, se debe reanudar el checkpoint usando `$env:LAB7_DEVICE="cpu"`; no se duplica el corpus ni se cargan todos los pares en GPU.

## Reproducción

La semilla global es 42. El hardware y versiones se registran al inicio del notebook. SimLex-999 se usa únicamente después de fijar los modelos finales; AG News test sólo se evalúa al final.
