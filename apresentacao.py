"""
apresentacao.py - Reconhecimento em Tempo Real da Tela com Overlay Fixo (Always on Top)

Uso:
  python apresentacao.py

Teclas de controle:
  - 'q': encerra a aplicação
"""

import json
from collections import deque
import cv2
import mss
import numpy as np
import tensorflow as tf
from utils import letterbox

CLASSE_VAZIA = "fundo"
LIMIAR = 0.80             # Limiar calibrado que eliminou falsos positivos
JANELA_SUAVIZACAO = 12    # Suavização por média móvel estável

print("Carregando modelo e configurações...")
model = tf.keras.models.load_model("modelo.keras")
class_names = json.load(open("class_names.json"))

fila_probs = deque(maxlen=JANELA_SUAVIZACAO)
sct = mss.mss()
monitor = sct.monitors[1]  # Captura o monitor principal inteiro

JANELA_NOME = "Pokemon HUD - Tempo Real"
cv2.namedWindow(JANELA_NOME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(JANELA_NOME, 420, 260)
# Mantém a janela sempre por cima de outros programas (Always on Top)
cv2.setWindowProperty(JANELA_NOME, cv2.WND_PROP_TOPMOST, 1)

print(f"Capturando monitor: {monitor['width']}x{monitor['height']}")
print("Pressione 'q' na janela do HUD para encerrar.")

while True:
    # 1. Captura da tela
    sct_img = sct.grab(monitor)
    frame = np.array(sct_img)[:, :, :3]  # Remove canal alpha, mantém BGR

    # 2. Pré-processamento idêntico ao treino (letterbox 224x224)
    img_rgb = cv2.cvtColor(letterbox(frame), cv2.COLOR_BGR2RGB).astype("float32")

    # 3. Inferência e Suavização Temporal
    probs = model.predict(img_rgb[None], verbose=0)[0]
    fila_probs.append(probs)
    probs_media = np.mean(fila_probs, axis=0)

    idx_max = int(np.argmax(probs_media))
    prob_max = probs_media[idx_max]
    rotulo = class_names[idx_max]

    # 4. Regras de Limiar e Fundo
    if prob_max < LIMIAR:
        texto = "?"
        cor = (0, 165, 255)  # Laranja
    elif rotulo == CLASSE_VAZIA:
        texto = "Nenhum objeto"
        cor = (0, 200, 255)  # Amarelo
    else:
        texto = rotulo.upper()
        cor = (0, 220, 0)    # Verde

    # 5. Desenho do Painel HUD Compacto (420x260)
    hud = np.full((260, 420, 3), 28, dtype=np.uint8)

    # Cabeçalho com o status do Pokémon
    cv2.rectangle(hud, (0, 0), (420, 60), (40, 40, 40), -1)
    cv2.putText(hud, f"{texto} ({prob_max:.0%})", (15, 42),
                cv2.FONT_HERSHEY_DUPLEX, 0.95, cor, 2)

    # Barras de probabilidade de cada classe
    for j, nome in enumerate(class_names):
        y = 90 + j * 38
        p = probs_media[j]
        largura_barra = int(p * 200)

        # Trilho cinza
        cv2.rectangle(hud, (15, y), (215, y + 18), (60, 60, 60), -1)
        # Barra colorida preenchida (destaca a classe vencedora)
        cor_barra = cor if j == idx_max and texto != "?" else (170, 170, 170)
        cv2.rectangle(hud, (15, y), (15 + largura_barra, y + 18), cor_barra, -1)
        # Rótulo e porcentagem
        cv2.putText(hud, f"{nome:<10} {p:>5.1%}", (225, y + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (230, 230, 230), 1)

    cv2.imshow(JANELA_NOME, hud)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()

