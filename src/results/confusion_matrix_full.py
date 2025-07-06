import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Tuple

def analisar_e_salvar_imagem(resultados: List[Tuple[str, str]]):
    """
    Recebe uma lista de tuplas, realiza a análise completa, imprime um resumo
    no console e salva uma imagem da matriz de confusão detalhada.

    Args:
        resultados: Uma lista de tuplas no formato ('Ação do LLM', 'Avaliação').
    """
    if not resultados:
        print("A lista de resultados está vazia.")
        return

    # 1. Criar o DataFrame e a matriz de contagens com crosstab
    colunas = ['LLM forneceu resposta', 'Resposta é util?']
    df = pd.DataFrame(resultados, columns=colunas)

    matriz_detalhada = pd.crosstab(
        df['Resposta é util?'],
        df['LLM forneceu resposta']
    )
    
    # Reordenar as linhas para uma ordem lógica e garantir que todas existam
    ordem_linhas = ['SIM', 'NÃO', 'NÃO_CONSTA_NOS_DOCS']
    matriz_detalhada = matriz_detalhada.reindex(ordem_linhas, fill_value=0)
    
    # Função auxiliar para pegar valores da matriz de forma segura
    def get_value(row, col):
        try:
            return matriz_detalhada.loc[row, col]
        except KeyError:
            return 0

    # 2. Extrair os valores de cada comportamento
    vp = get_value('SIM', 'SIM')
    vn = get_value('NÃO_CONSTA_NOS_DOCS', 'NÃO')
    fp_errada = get_value('NÃO', 'SIM')
    fp_alucinacao = get_value('NÃO_CONSTA_NOS_DOCS', 'SIM')
    fn = get_value('NÃO', 'NÃO') + get_value('SIM', 'NÃO')

    # 3. Imprimir o resumo dos componentes
    print("--- Resumo dos Comportamentos ---")
    print(f"✅ Respostas Corretas (VP): {vp}")
    # ... (o resto das impressões foi omitido para brevidade) ...

    # ... (cálculo das métricas omitido para brevidade) ...
    
    # 4. GERAR E SALVAR A IMAGEM DA MATRIZ
    plt.figure(figsize=(10, 7))
    heatmap = sns.heatmap(
        matriz_detalhada, 
        annot=True,         # Escreve os números dentro das células
        fmt='d',            # Formata os números como inteiros
        cmap='Greens',        # Paleta de cores
        linewidths=.5,      # Linhas de borda entre as células
        annot_kws={"size": 16} # Tamanho da fonte dos números
    )
    
    plt.title('Matriz de Confusão Detalhada do LLM', fontsize=18, pad=20)
    plt.ylabel('Real (Avaliação da Resposta)', fontsize=14)
    plt.xlabel('Previsto (Ação do LLM)', fontsize=14)
    plt.xticks(rotation=0, ha='center')
    plt.yticks(rotation=0)
    
    # Adicionar anotações de comportamento em cada célula
    # As coordenadas são (coluna, linha) + 0.5 para centralizar
    heatmap.text(1.5, 0.5, f'VP = {vp}', ha='center', va='center', color='white', fontsize=12)
    heatmap.text(0.5, 2.5, f'VN = {vn}', ha='center', va='center', color='white', fontsize=12)
    heatmap.text(1.5, 1.5, f'FP (Erro) = {fp_errada}', ha='center', va='center', color='black', fontsize=12)
    heatmap.text(1.5, 2.5, f'FP (Aluc.) = {fp_alucinacao}', ha='center', va='center', color='white', fontsize=12)
    heatmap.text(0.5, 1.5, f'FN (Omissão) = {fn}', ha='center', va='center', color='black', fontsize=12)
    
    nome_arquivo = 'matriz_detalhada_visual.png'
    plt.savefig(nome_arquivo, bbox_inches='tight')
    print(f"\nImagem da matriz salva com sucesso como '{nome_arquivo}'")


# --- Exemplo de Uso ---
if __name__ == '__main__':
    # Lista de resultados para análise
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

    # Chama a função para analisar os resultados e imprimir o resumo
    analisar_e_salvar_imagem(lista_de_resultados_exemplo)