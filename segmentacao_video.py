import cv2
import numpy as np
import keras_hub


# ============================================================
# CONFIGURAÇÕES
# ============================================================

VIDEO_ENTRADA = "video1.mp4"
VIDEO_SAIDA = "resultado_segmentacao.mp4"

TAMANHO_MODELO = 512

# Pascal VOC:
# classe 0 = background
# classe 15 = person
CLASSE_PESSOA = 15


# ============================================================
# CARREGAR DEEPLABV3+
# ============================================================

print("Carregando DeepLabV3+...")

modelo = keras_hub.models.ImageSegmenter.from_preset(
    "deeplab_v3_plus_resnet50_pascalvoc"
)

print("Modelo carregado!")


# ============================================================
# ABRIR VÍDEO
# ============================================================

cap = cv2.VideoCapture(VIDEO_ENTRADA)

if not cap.isOpened():
    raise FileNotFoundError(
        f"Não foi possível abrir o vídeo: {VIDEO_ENTRADA}"
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


    # --------------------------------------------------------
    # OpenCV usa BGR.
    # O modelo recebe RGB.
    # --------------------------------------------------------

    frame_rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # --------------------------------------------------------
    # Redimensionar para o modelo
    # --------------------------------------------------------

    imagem_modelo = cv2.resize(
        frame_rgb,
        (TAMANHO_MODELO, TAMANHO_MODELO),
        interpolation=cv2.INTER_LINEAR
    )


    # --------------------------------------------------------
    # Criar dimensão de batch
    #
    # (512,512,3)
    #       ↓
    # (1,512,512,3)
    # --------------------------------------------------------

    entrada = np.expand_dims(
        imagem_modelo,
        axis=0
    )


    # --------------------------------------------------------
    # SEGMENTAÇÃO
    # --------------------------------------------------------

    predicao = modelo.predict(
        entrada,
        verbose=0
    )


    # --------------------------------------------------------
    # Para cada pixel, pega a classe com maior probabilidade
    # --------------------------------------------------------

    mapa_classes = np.argmax(
        predicao,
        axis=-1
    )[0]


    # --------------------------------------------------------
    # Criar máscara apenas da classe PERSON
    # --------------------------------------------------------

    mascara = (
        mapa_classes == CLASSE_PESSOA
    ).astype(np.uint8)


    # --------------------------------------------------------
    # Voltar máscara ao tamanho original
    # --------------------------------------------------------

    mascara = cv2.resize(
        mascara,
        (largura, altura),
        interpolation=cv2.INTER_NEAREST
    )


    # ========================================================
    # SOBREPOSIÇÃO VERDE
    # ========================================================

    overlay = frame.copy()

    overlay[
        mascara == 1
    ] = (0, 255, 0)


    resultado = cv2.addWeighted(
        frame,
        0.65,
        overlay,
        0.35,
        0
    )


    # ========================================================
    # TEXTO
    # ========================================================

    cv2.putText(
        resultado,
        "DeepLabV3+ - Pessoa",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )


    # ========================================================
    # SALVAR
    # ========================================================

    saida.write(resultado)


    # ========================================================
    # EXIBIR MENOR NA TELA
    # ========================================================

    visualizacao = cv2.resize(
        resultado,
        None,
        fx=0.5,
        fy=0.5
    )

    cv2.imshow(
        "Segmentacao - DeepLabV3+",
        visualizacao
    )


    # Q para sair
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ============================================================
# FINALIZAR
# ============================================================

cap.release()
saida.release()

cv2.destroyAllWindows()

print("\nProcessamento finalizado!")
print(f"Vídeo salvo em: {VIDEO_SAIDA}")