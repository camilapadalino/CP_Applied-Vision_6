import cv2
import numpy as np
import keras_hub
from ultralytics import YOLO


# ============================================================
# CONFIGURAÇÕES
# ============================================================

VIDEO_ENTRADA = "video1.mp4"
VIDEO_SAIDA = "resultado_yolo_segmentacao.mp4"

CONFIANCA_YOLO = 0.15

# Pascal VOC:
# 0 = background
# 15 = person
CLASSE_PESSOA = 15

TAMANHO_SEGMENTACAO = 512


# ============================================================
# CARREGAR MODELOS
# ============================================================

print("Carregando YOLO...")

modelo_yolo = YOLO("yolo11n.pt")

print("Carregando DeepLabV3+...")

modelo_segmentacao = (
    keras_hub.models.ImageSegmenter.from_preset(
        "deeplab_v3_plus_resnet50_pascalvoc"
    )
)

print("Modelos carregados!")


# ============================================================
# ABRIR VÍDEO
# ============================================================

cap = cv2.VideoCapture(VIDEO_ENTRADA)

if not cap.isOpened():
    raise FileNotFoundError(
        f"Não foi possível abrir: {VIDEO_ENTRADA}"
    )


fps = cap.get(cv2.CAP_PROP_FPS)

largura = int(
    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
)

altura = int(
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
)


# ============================================================
# CRIAR VÍDEO DE SAÍDA
# ============================================================

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

saida = cv2.VideoWriter(
    VIDEO_SAIDA,
    fourcc,
    fps,
    (largura, altura)
)


# ============================================================
# PROCESSAMENTO
# ============================================================

print("Processando vídeo...")


while True:

    sucesso, frame = cap.read()

    if not sucesso:
        break


    # ========================================================
    # 1. YOLO
    # ========================================================

    resultados = modelo_yolo.predict(
        source=frame,
        conf=CONFIANCA_YOLO,
        verbose=False
    )

    resultado = resultados[0]


    # ========================================================
    # 2. PERCORRER DETECÇÕES
    # ========================================================

    for caixa in resultado.boxes:

        classe_id = int(
            caixa.cls.item()
        )

        nome_classe = (
            modelo_yolo.names[
                classe_id
            ]
        )


        # Queremos somente pessoas
        if nome_classe != "person":
            continue


        confianca = float(
            caixa.conf.item()
        )


        x1, y1, x2, y2 = map(
            int,
            caixa.xyxy[0].tolist()
        )


        # Evita coordenadas fora da imagem
        x1 = max(0, x1)
        y1 = max(0, y1)

        x2 = min(largura, x2)
        y2 = min(altura, y2)


        # ====================================================
        # 3. RECORTAR A PESSOA
        # ====================================================

        pessoa = frame[
            y1:y2,
            x1:x2
        ]


        if pessoa.size == 0:
            continue


        altura_recorte = (
            pessoa.shape[0]
        )

        largura_recorte = (
            pessoa.shape[1]
        )


        # ====================================================
        # 4. PREPARAR PARA DEEPLABV3+
        # ====================================================

        pessoa_rgb = cv2.cvtColor(
            pessoa,
            cv2.COLOR_BGR2RGB
        )


        pessoa_modelo = cv2.resize(
            pessoa_rgb,
            (
                TAMANHO_SEGMENTACAO,
                TAMANHO_SEGMENTACAO
            ),
            interpolation=cv2.INTER_LINEAR
        )


        entrada = np.expand_dims(
            pessoa_modelo,
            axis=0
        )


        # ====================================================
        # 5. SEGMENTAÇÃO
        # ====================================================

        predicao = (
            modelo_segmentacao.predict(
                entrada,
                verbose=0
            )
        )


        mapa_classes = np.argmax(
            predicao,
            axis=-1
        )[0]


        # Apenas pixels classificados
        # como PERSON
        mascara = (
            mapa_classes
            == CLASSE_PESSOA
        ).astype(np.uint8)


        # ====================================================
        # 6. VOLTAR AO TAMANHO DO RECORTE
        # ====================================================

        mascara = cv2.resize(
            mascara,
            (
                largura_recorte,
                altura_recorte
            ),
            interpolation=cv2.INTER_NEAREST
        )


        # ====================================================
        # 7. PROJETAR MÁSCARA NO FRAME
        # ====================================================

        regiao_frame = frame[
            y1:y2,
            x1:x2
        ]


        overlay = (
            regiao_frame.copy()
        )


        overlay[
            mascara == 1
        ] = (
            0,
            255,
            0
        )


        regiao_resultado = (
            cv2.addWeighted(
                regiao_frame,
                0.65,
                overlay,
                0.35,
                0
            )
        )


        frame[
            y1:y2,
            x1:x2
        ] = regiao_resultado


        # ====================================================
        # 8. BOUNDING BOX DO YOLO
        # ====================================================

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            3
        )


        texto = (
            f"Pessoa: "
            f"{confianca * 100:.1f}%"
        )


        cv2.putText(
            frame,
            texto,
            (
                x1,
                max(y1 - 10, 25)
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )


    # ========================================================
    # IDENTIFICAÇÃO DO MODELO
    # ========================================================

    cv2.putText(
        frame,
        "YOLO + DeepLabV3+",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )


    # ========================================================
    # SALVAR
    # ========================================================

    saida.write(frame)


    # ========================================================
    # MOSTRAR NA TELA
    # ========================================================

    visualizacao = cv2.resize(
        frame,
        None,
        fx=0.5,
        fy=0.5
    )


    cv2.imshow(
        "YOLO + Segmentacao",
        visualizacao
    )


    # Aperte Q para sair
    if (
        cv2.waitKey(1) & 0xFF
        == ord("q")
    ):
        break


# ============================================================
# FINALIZAR
# ============================================================

cap.release()
saida.release()

cv2.destroyAllWindows()


print()
print("==============================")
print("PROCESSAMENTO FINALIZADO")
print("==============================")
print(
    f"Vídeo salvo em: "
    f"{VIDEO_SAIDA}"
)