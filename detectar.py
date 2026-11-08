"""
Passo 3 - Reconhece objetos em um video com suavizacao e gera a linha do tempo.
Uso: python detectar.py teste_final.mp4
"""
import sys
import os
import json
from collections import deque
import cv2
import numpy as np
import tensorflow as tf
from utils import letterbox

CLASSE_VAZIA = "fundo"
JANELA_SUAVIZACAO = 12
LIMIAR = 0.80

model = tf.keras.models.load_model("modelo.keras")
class_names = json.load(open("class_names.json"))

fonte = sys.argv[1] if len(sys.argv) > 1 else "0"
webcam = fonte.isdigit()
cap = cv2.VideoCapture(int(fonte) if webcam else fonte)
if not cap.isOpened():
    raise SystemExit(f"Nao consegui abrir: {fonte}")

fps = cap.get(cv2.CAP_PROP_FPS) or 30
writer = None
if not webcam:
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    saida = f"saida_{os.path.splitext(os.path.basename(fonte))[0]}.mp4"
    writer = cv2.VideoWriter(saida, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

historico_probs = deque(maxlen=JANELA_SUAVIZACAO)
rotulos_por_frame = []
n = 0

while True:
    ok, frame = cap.read()
    if not ok:
        break
    if webcam:
        frame = cv2.flip(frame, 1)

    img = cv2.cvtColor(letterbox(frame), cv2.COLOR_BGR2RGB).astype("float32")
    raw_probs = model.predict(img[None], verbose=0)[0]

    # 1. Suavizacao Temporal
    historico_probs.append(raw_probs)
    probs_suavizadas = np.mean(historico_probs, axis=0)

    i = int(np.argmax(probs_suavizadas))
    maior_conf = float(probs_suavizadas[i])

    # 2. Limiar de Confianca
    if maior_conf < LIMIAR:
        rotulo = "?"
    else:
        rotulo = class_names[i]

    rotulos_por_frame.append(rotulo)

    # Desenho
    if rotulo == "?":
        texto = "?"
        cor = (0, 140, 255)
    elif rotulo == CLASSE_VAZIA:
        texto = "nenhum objeto"
        cor = (0, 200, 255)
    else:
        texto = rotulo
        cor = (0, 200, 0)

    cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (0, 0, 0), -1)
    cv2.putText(frame, f"{texto} ({maior_conf:.0%})", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, cor, 2)

    for j, c in enumerate(class_names):
        y = 55 + j * 22
        cv2.rectangle(frame, (10, y), (10 + int(probs_suavizadas[j] * 150), y + 14), (0, 200, 255), -1)
        cv2.putText(frame, c, (170, y + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.putText(frame, f"t = {n / fps:5.1f}s", (10, frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    if writer is not None:
        writer.write(frame)
    if webcam or os.environ.get("MOSTRAR", "1") == "1":
        cv2.imshow("Reconhecimento com CNN", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    n += 1

cap.release()
if writer is not None:
    writer.release()
cv2.destroyAllWindows()

# 3. Linha do Tempo
print("\n" + "=" * 45)
print("Linha do tempo (Pokemon identificados):")
print("=" * 45)

trechos = []
if rotulos_por_frame:
    rotulo_atual = rotulos_por_frame[0]
    frame_inicio = 0
    for idx, r in enumerate(rotulos_por_frame):
        if r != rotulo_atual:
            duracao = (idx - frame_inicio) / fps
            if duracao >= 0.5 and rotulo_atual not in (CLASSE_VAZIA, "?"):
                t_ini = frame_inicio / fps
                t_fim = idx / fps
                trechos.append((t_ini, t_fim, rotulo_atual))
            rotulo_atual = r
            frame_inicio = idx
    # Ultimo segmento
    duracao = (len(rotulos_por_frame) - frame_inicio) / fps
    if duracao >= 0.5 and rotulo_atual not in (CLASSE_VAZIA, "?"):
        t_ini = frame_inicio / fps
        t_fim = len(rotulos_por_frame) / fps
        trechos.append((t_ini, t_fim, rotulo_atual))

encontrados = set()
for t_ini, t_fim, pkm in trechos:
    print(f"{t_ini:5.1f}s a {t_fim:5.1f}s -> {pkm}")
    encontrados.add(pkm)

print("\nPokemon encontrados no video:", sorted(list(encontrados)))
if writer is not None:
    print(f"\nVideo anotado salvo em: {saida}")

