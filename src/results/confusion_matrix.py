
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay

# Definindo as strings padronizadas para robustez
LLM_RESPONDEU_SIM = "SIM"
LLM_RESPONDEU_NAO = "NÃO"
AVALIACAO_SIM = "SIM"
AVALIACAO_NAO = "NÃO"
AVALIACAO_NAO_CONSTA = "NÃO_CONSTA_NOS_DOCS"


def gerar_matriz_de_confusao(resultados: list[tuple[str, str]]):
    """
    Recebe uma lista de tuplas e gera uma matriz de confusão 2x2 completa,
    incluindo a possibilidade de Falsos Negativos (Omissões).

    Args:
        resultados: Uma lista de tuplas no formato ('Ação do LLM', 'Avaliação').
    """
    if not resultados:
        print("A lista de resultados está vazia.")
        return

    # 1. Criar DataFrame a partir dos resultados
    colunas = ["LLM forneceu resposta", "Resposta é util?"]
    df = pd.DataFrame(resultados, columns=colunas)

    # 2. Calcular os valores para cada quadrante da matriz

    # VP (Verdadeiro Positivo): LLM respondeu SIM e a avaliação foi SIM
    vp = df[
        (df["LLM forneceu resposta"] == LLM_RESPONDEU_SIM)
        & (df["Resposta é util?"] == AVALIACAO_SIM)
    ].shape[0]

    # VN (Verdadeiro Negativo): LLM respondeu NÃO e a avaliação foi NÃO_CONSTA_NOS_DOCS
    vn = df[
        (df["LLM forneceu resposta"] == LLM_RESPONDEU_NAO)
        & (df["Resposta é util?"] == AVALIACAO_NAO_CONSTA)
    ].shape[0]

    # FP (Falso Positivo): Agrupa Alucinações e Respostas Erradas
    fp_alucinacao = df[
        (df["LLM forneceu resposta"] == LLM_RESPONDEU_SIM)
        & (df["Resposta é util?"] == AVALIACAO_NAO_CONSTA)
    ].shape[0]
    fp_resposta_errada = df[
        (df["LLM forneceu resposta"] == LLM_RESPONDEU_SIM)
        & (df["Resposta é util?"] == AVALIACAO_NAO)
    ].shape[0]
    fp = fp_alucinacao + fp_resposta_errada

    # FN (Falso Negativo): Omissões. LLM respondeu NÃO, mas deveria ter respondido.
    # Esta é a principal alteração: FN não é mais sempre zero.
    fn = df[
        (df["LLM forneceu resposta"] == LLM_RESPONDEU_NAO)
        & (df["Resposta é util?"] == AVALIACAO_NAO)
    ].shape[0]

    print("--- Contagem para a Matriz de Confusão Completa ---")
    print(f"Verdadeiros Positivos (VP - Respostas Corretas): {vp}")
    print(f"Verdadeiros Negativos (VN - Recusas Corretas): {vn}")
    print(
        f"Falsos Positivos (FP - Alucinações + Respostas Erradas): {fp} ({fp_alucinacao} aluc. + {fp_resposta_errada} err.)"
    )
    print(f"Falsos Negativos (FN - Omissões): {fn}")
    print("-" * 50)

    # 3. Montar e exibir a matriz
    matriz = np.array([[vn, fp], [fn, vp]])

    display_labels_previsto = ["NÃO Respondeu", "SIM Respondeu"]
    display_labels_real = ["NÃO Deveria Responder", "SIM Deveria Responder"]

    disp = ConfusionMatrixDisplay(confusion_matrix=matriz, display_labels=display_labels_previsto)

    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap=plt.cm.Blues, values_format="d")
    ax.set_xticklabels(display_labels_previsto)
    ax.set_yticklabels(display_labels_real, rotation=90, va="center")
    plt.title("Matriz de Confusão da Avaliação do LLM", fontsize=16)
    plt.xlabel("O que o LLM Fez (Previsto)")
    plt.ylabel("O que Deveria Acontecer (Real)")

    nome_arquivo = "matriz_de_confusao_completa.png"
    plt.savefig(nome_arquivo, bbox_inches="tight")
    print(f"\nGráfico '{nome_arquivo}' foi salvo no diretório atual.")


# --- COMO USAR A FUNÇÃO ---
if __name__ == "__main__":
    # Substitua esta lista de exemplo pelos seus dados reais.
    # Formato: (LLM forneceu resposta, Resposta é util?)
    lista_de_resultados_exemplo = [
        ("SIM", "NÃO"),
        ("SIM", "NÃO"),
        ("NÃO", "NÃO_CONSTA_NOS_DOCS"),
        ("SIM", "SIM"),
        ("NÃO", "NÃO_CONSTA_NOS_DOCS"),
        ("NÃO", "NÃO_CONSTA_NOS_DOCS"),
        ("NÃO", "NÃO_CONSTA_NOS_DOCS"),
        ("SIM", "SIM"),
        ("SIM", "NÃO"),
        ("SIM", "SIM"),
        ("SIM", "SIM"),
        ("SIM", "SIM"),
        ("SIM", "NÃO"),
        ("SIM", "SIM"),
        ("SIM", "SIM"),
        ("SIM", "SIM"),
        ("SIM", "SIM"),
        ("SIM", "SIM"),
        ("SIM", "SIM"),
        ("NÃO", "NÃO_CONSTA_NOS_DOCS"),
        ("SIM", "SIM"),
        ("SIM", "SIM"),
        ("SIM", "SIM"),
        ("SIM", "SIM"),
        ("SIM", "SIM"),
        ("SIM", "SIM"),
        ("NÃO", "NÃO_CONSTA_NOS_DOCS"),
        ("SIM", "SIM"),
        ("SIM", "SIM"),
        ("NÃO", "NÃO_CONSTA_NOS_DOCS"),
        ("NÃO", "NÃO_CONSTA_NOS_DOCS"),
        ("SIM", "SIM"),
        ("SIM", "SIM"),
        ("NÃO", "NÃO_CONSTA_NOS_DOCS"),
        ("SIM", "SIM"),
        ("SIM", "SIM"),
        ("SIM", "SIM"),
        ("SIM", "SIM"),
        ("SIM", "SIM"),
        ("SIM", "SIM"),
    ]

    # Chama a função principal para rodar a análise
    gerar_matriz_de_confusao(lista_de_resultados_exemplo)
