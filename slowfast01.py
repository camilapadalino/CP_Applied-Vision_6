import json
import urllib.request
import cv2
import numpy as np
import torch


# ============================================================
# CONFIGURAÇÕES
# ============================================================

ARQUIVO_VIDEO = "video1.mp4"

NUMERO_QUADROS = 32
TAMANHO_IMAGEM = 224
ALPHA = 4

ARQUIVO_CLASSES = "kinetics_classnames.json"

URL_CLASSES = (
    "https://dl.fbaipublicfiles.com/"
    "pyslowfast/dataset/class_names/"
    "kinetics_classnames.json"
)


# ============================================================
# DISPOSITIVO
# ============================================================

dispositivo = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Dispositivo:", dispositivo)


# ============================================================
# CLASSES KINETICS
# ============================================================

def carregar_classes():

    try:

        with open(
            ARQUIVO_CLASSES,
            "r",
            encoding="utf-8"
        ) as arquivo:

            classes = json.load(arquivo)

    except FileNotFoundError:

        print("Baixando os nomes das classes Kinetics-400...")

        urllib.request.urlretrieve(
            URL_CLASSES,
            ARQUIVO_CLASSES
        )

        with open(
            ARQUIVO_CLASSES,
            "r",
            encoding="utf-8"
        ) as arquivo:

            classes = json.load(arquivo)

    id_para_classe = {}

    for nome, identificador in classes.items():

        id_para_classe[int(identificador)] = nome

    return id_para_classe


# ============================================================
# CARREGAR VÍDEO
# ============================================================

def carregar_video(caminho):

    cap = cv2.VideoCapture(caminho)

    if not cap.isOpened():

        raise FileNotFoundError(
            f"Não foi possível abrir: {caminho}"
        )

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    indices = np.linspace(
        0,
        total_frames - 1,
        NUMERO_QUADROS
    ).astype(int)

    frames = []

    for indice in indices:

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            indice
        )

        sucesso, frame = cap.read()

        if not sucesso:
            continue

        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        frame = cv2.resize(
            frame,
            (
                TAMANHO_IMAGEM,
                TAMANHO_IMAGEM
            )
        )

        frames.append(frame)

    cap.release()

    if len(frames) != NUMERO_QUADROS:

        raise RuntimeError(
            "Não foi possível carregar "
            "os 32 frames necessários."
        )

    video = np.stack(frames)

    # T,H,W,C -> C,T,H,W
    video = torch.from_numpy(
        video
    ).permute(
        3, 0, 1, 2
    )

    video = video.float() / 255.0

    return video


# ============================================================
# NORMALIZAÇÃO
# ============================================================

def normalizar(video):

    media = torch.tensor(
        [0.45, 0.45, 0.45]
    ).view(
        3, 1, 1, 1
    )

    desvio = torch.tensor(
        [0.225, 0.225, 0.225]
    ).view(
        3, 1, 1, 1
    )

    return (
        video - media
    ) / desvio


# ============================================================
# SLOW + FAST
# ============================================================

def criar_caminhos(video):

    # Fast utiliza todos os frames
    fast = video

    quantidade_slow = (
        video.shape[1] // ALPHA
    )

    indices_slow = torch.linspace(
        0,
        video.shape[1] - 1,
        quantidade_slow
    ).long()

    slow = torch.index_select(
        video,
        1,
        indices_slow
    )

    slow = slow.unsqueeze(0).to(
        dispositivo
    )

    fast = fast.unsqueeze(0).to(
        dispositivo
    )

    return [slow, fast]


# ============================================================
# MODELO
# ============================================================

print("Carregando SlowFast...")

modelo = torch.hub.load(
    "facebookresearch/pytorchvideo",
    "slowfast_r50",
    pretrained=True
)

modelo = modelo.to(dispositivo)

modelo.eval()


# ============================================================
# PROCESSAMENTO
# ============================================================

classes = carregar_classes()

video = carregar_video(
    ARQUIVO_VIDEO
)

video = normalizar(
    video
)

caminhos = criar_caminhos(
    video
)


# ============================================================
# CLASSIFICAÇÃO
# ============================================================

print("\nClassificando ação...\n")

with torch.no_grad():

    resultado = modelo(
        caminhos
    )

probabilidades = torch.softmax(
    resultado,
    dim=1
)

valores, indices = torch.topk(
    probabilidades,
    k=5,
    dim=1
)


# ============================================================
# RESULTADOS
# ============================================================

print("TOP 5 AÇÕES:\n")

for posicao in range(5):

    indice = int(
        indices[0][posicao].item()
    )

    confianca = float(
        valores[0][posicao].item()
    )

    nome = classes.get(
        indice,
        f"Classe {indice}"
    )

    print(
        f"{posicao + 1}. "
        f"{nome}: "
        f"{confianca * 100:.2f}%"
    )