import cv2
from ultralytics import YOLO

VIDEO_ENTRADA = "video1.mp4"
VIDEO_SAIDA = "resultado_yolo.mp4"
CONFIANCA = 0.15

modelo = YOLO("yolo11n.pt")

cap = cv2.VideoCapture(VIDEO_ENTRADA)

if not cap.isOpened():
    raise FileNotFoundError(
        f"Não foi possível abrir o vídeo: {VIDEO_ENTRADA}"
    )

fps = cap.get(cv2.CAP_PROP_FPS)
largura = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
altura = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

saida = cv2.VideoWriter(
    VIDEO_SAIDA,
    fourcc,
    fps,
    (largura, altura)
)

while True:

    sucesso, frame = cap.read()

    if not sucesso:
        break

    resultados = modelo.predict(
        source=frame,
        conf=CONFIANCA,
        verbose=False
    )

    resultado = resultados[0]

    for caixa in resultado.boxes:

        classe_id = int(caixa.cls.item())
        nome_classe = modelo.names[classe_id]

        # Apenas pessoas
        if nome_classe != "person":
            continue

        confianca = float(caixa.conf.item())

        x1, y1, x2, y2 = map(
            int,
            caixa.xyxy[0].tolist()
        )

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            4
        )

        texto = f"Pessoa: {confianca * 100:.1f}%"

        cv2.putText(
            frame,
            texto,
            (x1, max(y1 - 10, 25)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    saida.write(frame)

    # Só diminui a visualização, não o vídeo salvo
    visualizacao = cv2.resize(
        frame,
        None,
        fx=0.5,
        fy=0.5
    )

    cv2.imshow(
        "YOLO - Deteccao de Pessoas",
        visualizacao
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
saida.release()
cv2.destroyAllWindows()

print(f"Vídeo salvo em: {VIDEO_SAIDA}")