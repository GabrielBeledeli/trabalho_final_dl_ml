# Reconhecimento de Pokémon em Vídeo com Deep Learning
### Pipeline Completo com MobileNetV2 (Transfer Learning & Fine-Tuning)

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-MobileNetV2-D00000?style=for-the-badge&logo=keras&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-Video_HUD-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)

Pipeline completo de **Visão Computacional e Deep Learning** para extração de quadros, treinamento em duas fases (**MobileNetV2** com *Transfer Learning* e *Fine-Tuning*) e reconhecimento estabilizado de objetos em vídeo, webcam ou captura de tela em tempo real.

O repositório **já inclui os vídeos base de gravação (`videos/`) e um vídeo de validação (`teste_final.mp4`)** com **Bulbasaur**, **Charmander**, **Pikachu** e **Fundo**, permitindo que qualquer pessoa clone o projeto e rode todo o fluxo do zero (extração → treino → detecção) imediatamente — ou substitua os vídeos para treinar suas próprias classes.

---

## O que este projeto faz?

1. **Extrai e padroniza frames automaticamente (`extrair_frames.py`):** Converte os vídeos de `videos/` em imagens `224x224` preservando a proporção com bordas pretas (`letterbox`), filtrando quadros consecutivos quase idênticos.
2. **Treina uma CNN sem vazamento de dados (`train.py`):** Separa treino e validação **por vídeo inteiro** (*No Data Leakage*), executa *Transfer Learning* + *Fine-Tuning* na MobileNetV2 e gera o modelo (`modelo.keras`), gráfico de curvas (`curvas.png`), matriz de confusão e catálogo em PDF (`dataset_imagens.pdf`).
3. **Detecta em vídeo ou webcam com Linha do Tempo (`detectar.py`):** Aplica suavização temporal por média móvel (`deque`) e limiar de confiança (`80%`), grava o vídeo anotado com HUD (`saida_teste_final.mp4`) e exibe no terminal em quais segundos cada Pokémon apareceu.
4. **Reconhece a tela em tempo real (`apresentacao.py`):** Abre um mini-painel HUD fixo sempre no topo (*Always on Top*) que classifica ao vivo imagens, vídeos ou modelos 3D abertos no monitor.

---

## Estrutura Inicial do Repositório

Ao clonar o projeto, você já terá todos os scripts e os vídeos de entrada prontos para rodar o passo a passo:

```text
.
├── videos/                 # Vídeos brutos de entrada (4 clipes por classe)
│   ├── bulbasaur/          # v1.mp4, v2.mp4, v3.mp4, v4.mp4
│   ├── charmander/         # v1.mp4, v2.mp4, v3.mp4, v4.mp4
│   ├── pikachu/            # v1.mp4, v2.mp4, v3.mp4, v4.mp4
│   └── fundo/              # v1.mp4, v2.mp4, v3.mp4, v4.mp4 (cenário vazio)
├── teste_final.mp4         # Vídeo contínuo de teste alternando os 3 Pokémon e fundo
├── utils.py                # Pré-processamento compartilhado (função letterbox 224x224)
├── extrair_frames.py       # Passo 1: Gera a pasta data/ a partir de videos/
├── train.py                # Passo 2: Treina o modelo e gera modelo.keras, curvas.png e PDF
├── detectar.py             # Passo 3: Roda inferência no vídeo/webcam e gera linha do tempo
├── apresentacao.py         # Extra: Overlay (Always on Top) capturando o monitor ao vivo
├── requirements.txt        # Dependências do ambiente Python
└── .gitignore              # Mantém fora do Git os arquivos gerados localmente no treino
```

---

## Instalação e Configuração

Recomenda-se o uso do **Python 3.11+** dentro de um ambiente virtual (`venv`).

### 1. Clonar o repositório
```bash
git clone <URL_DO_REPOSITORIO>
cd <NOME_DA_PASTA>
```

### 2. Criar a `venv` e instalar as dependências

**Windows (PowerShell):**
```powershell
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

*(Pacotes instalados: `tensorflow`, `numpy`, `matplotlib`, `opencv-python`, `pillow` e `mss`.)*

---

## Passo a Passo de Execução (Pronto para Rodar)

Como a pasta `videos/` e o `teste_final.mp4` já acompanham o repositório, basta executar os 3 passos em ordem:

### Passo 1 — Extrair os frames dos vídeos
```bash
python extrair_frames.py
```
- Lê os vídeos em `videos/<classe>/`, aplica `letterbox` (`224x224`) e cria localmente a pasta `data/` com os frames nomeados no padrão `<video>__<frame>.jpg` (ex.: `v1__000000.jpg`).
- Exibe no terminal a quantidade de frames extraídos de cada vídeo e o total por classe.

---

### Passo 2 — Treinar o modelo (`MobileNetV2`)
```bash
python train.py
```
- Separa automaticamente vídeos inteiros para validação (`20%`) e treino (`80%`), evitando vazamento temporal entre quadros vizinhos.
- Roda a **Fase 1** (base congelada, `lr = 1e-3`) seguida da **Fase 2** (*Fine-Tuning* descongelando as últimas 30 camadas, `lr = 1e-5`).
- Ao terminar, imprime a **Matriz de Confusão**, **Precisão**, **Recall** e **Acurácia de Validação**, salvando na raiz:
  - `modelo.keras` e `class_names.json`
  - `curvas.png` (gráfico de evolução do treino e validação)
  - `dataset_imagens.pdf` (grade visual com as fotos extraídas de cada classe)

---

### Passo 3 — Testar o reconhecimento no vídeo (`teste_final.mp4`)
```bash
python detectar.py teste_final.mp4
```
- Abre a visualização do vídeo com o **HUD superior** mostrando o Pokémon reconhecido (ou `"nenhum objeto"` para `fundo`, e `"?"` quando a confiança for menor que `80%`) e barras de probabilidade suavizadas em tempo real.
- Grava automaticamente o vídeo processado em `saida_teste_final.mp4`.
- Ao final do vídeo, imprime no terminal a **Linha do Tempo consolidada** com o segundo inicial e final de cada aparição e a lista dos Pokémon encontrados:

```text
=============================================
Linha do tempo (Pokemon identificados):
=============================================
  5.2s a  16.5s -> charmander
 24.7s a  39.6s -> bulbasaur
 45.8s a  59.0s -> pikachu

Pokemon encontrados no video: ['bulbasaur', 'charmander', 'pikachu']

Video anotado salvo em: saida_teste_final.mp4
```

*(Dica: pressione `q` na janela de vídeo caso queira encerrar antes do fim, ou defina `MOSTRAR=0` antes do comando para processar sem abrir janela gráfica.)*

---

## Modos Extras de Uso em Tempo Real

Depois de gerar o `modelo.keras` no Passo 2, você também pode testar ao vivo:

### Detecção via Webcam
```bash
python detectar.py 0
```

### Detecção da Tela em Tempo Real (*Always on Top*)
Ideal para abrir imagens em tela cheia, vídeos ou modelos 3D interativos (ex.: Sketchfab) no navegador enquanto um painel compacto (`420x260`) sobrepõe as barras de probabilidade em tempo real:
```bash
python apresentacao.py
```
*(Arraste o painel HUD para um canto do monitor e pressione `q` na janela do HUD para sair.)*

---

## Como Treinar com Seus Próprios Objetos ou Novos Pokémon

Se quiser usar o projeto para reconhecer outros objetos ou adicionar novos Pokémon:

1. Dentro de `videos/`, crie uma subpasta para cada nova classe (ou substitua as atuais) e mantenha sempre uma subpasta `fundo/` com vídeos apenas do ambiente vazio.
2. Grave pelo menos **3 a 4 vídeos curtos** (`10s` a `15s`) por classe variando ângulos e iluminação (`v1.mp4`, `v2.mp4`, `v3.mp4`, `v4.mp4`).
3. Apague a pasta `data/` antiga (se já tiver rodado antes) e execute novamente `python extrair_frames.py` → `python train.py` → `python detectar.py seu_video.mp4`.

### Principais Parâmetros Ajustáveis

| Arquivo | Parâmetro | Padrão | Função |
| :--- | :--- | :---: | :--- |
| `extrair_frames.py` | `FRAMES_POR_SEGUNDO` | `6` | Taxa de quadros avaliados por segundo de vídeo |
| `extrair_frames.py` | `DIFF_MINIMA` | `2.0` | Sensibilidade para descartar frames consecutivos muito parecidos |
| `extrair_frames.py` | `MAX_POR_VIDEO` | `120` | Teto de frames salvos por arquivo de vídeo |
| `train.py` | `EPOCHS_FASE1` / `EPOCHS_FASE2` | `8` / `6` | Épocas máximas do Transfer Learning e do Fine-Tuning |
| `detectar.py` / `apresentacao.py` | `JANELA_SUAVIZACAO` | `12` | Tamanho da janela da média móvel de probabilidades |
| `detectar.py` / `apresentacao.py` | `LIMIAR` | `0.80` | Confiança mínima (`80%`) para validar uma classe (abaixo exibe `"?"`) |

---

## Autores

- **Gabriel Beledeli Hul**
- **Nicolas Miguel**
- **Vinicius Buskievicz**
- **Alisson Eraldo**
