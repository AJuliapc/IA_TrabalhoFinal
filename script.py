import torch
import torch.nn as nn
import pandas as pd
import ltn
import random
import math
from pysat.formula import CNF
from pysat.solvers import Glucose3

# --- Questão 1: Classificação de Tabuleiro Sudoku Válido/Inválido ---

# ---------- Predicado "Different" ----------
class DifferentModel(nn.Module):
    def forward(self, x, y):
        return 1.0 - torch.exp(-100 * torch.abs(x - y))

Different = ltn.Predicate(DifferentModel())

# ---------- Carregar CSV ----------
def carregar_tabuleiro_csv_q1(path):
    df = pd.read_csv(path, header=None)
    return torch.tensor(df.values, dtype=torch.float32)

# ---------- Geração dos axiomas ----------
def gerar_axiomas(tabuleiro):
    n = tabuleiro.shape[0]
    axiomas = []

    # ---------- Linhas ----------
    for i in range(n):
        for j1 in range(n):
            for j2 in range(j1 + 1, n):
                v1_val = tabuleiro[i][j1].item()
                v2_val = tabuleiro[i][j2].item()
                if v1_val != 0 and v2_val != 0:
                    v1 = ltn.Constant(torch.tensor([v1_val]))
                    v2 = ltn.Constant(torch.tensor([v2_val]))
                    axiomas.append(Different(v1, v2))

    # ---------- Colunas ----------
    for j in range(n):
        for i1 in range(n):
            for i2 in range(i1 + 1, n):
                v1_val = tabuleiro[i1][j].item()
                v2_val = tabuleiro[i2][j].item()
                if v1_val != 0 and v2_val != 0:
                    v1 = ltn.Constant(torch.tensor([v1_val]))
                    v2 = ltn.Constant(torch.tensor([v2_val]))
                    axiomas.append(Different(v1, v2))

    # ---------- Blocos ----------
    b = int(n ** 0.5)
    for bi in range(0, n, b):
        for bj in range(0, n, b):
            celulas = []
            for i in range(bi, bi + b):
                for j in range(bj, bj + b):
                    val = tabuleiro[i][j].item()
                    if val != 0:
                        celulas.append(ltn.Constant(torch.tensor([val])))
            for k1 in range(len(celulas)):
                for k2 in range(k1 + 1, len(celulas)):
                    axiomas.append(Different(celulas[k1], celulas[k2]))

    return axiomas

# ---------- Classificador ----------
def classificar_tabuleiro(tabuleiro, limiar=0.95):
    axiomas = gerar_axiomas(tabuleiro)
    if not axiomas:
        return 1  # tabuleiro vazio é considerado válido
    verdades = torch.stack([ax.value for ax in axiomas]).squeeze()
    score = verdades.min()
    return 1 if score.item() >= limiar else 0

# --- Questão 2: Previsão de Jogadas Válidas com LTN e MLP ---

# ---------- Modelo MLP treinável ----------
class ValidMoveModel(nn.Module):
    def __init__(self, n=4):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.mlp(x)

# Predicado LTN (instanciado globalmente para o treinamento)
model_q2 = ValidMoveModel()
ValidMove = ltn.Predicate(model_q2)

# ---------- Carrega tabuleiro (reutiliza para Q2 e Q3) ----------
def carregar_tabuleiro_csv_q2_q3(path):
    df = pd.read_csv(path, header=None)
    return torch.tensor(df.values, dtype=torch.float32)

# ---------- Verifica se (i,j,v) é válida ----------
def jogada_valida(tabuleiro, i, j, v):
    n = tabuleiro.shape[0]
    # Verifica linha
    if v in tabuleiro[i] :
        return False
    # Verifica coluna
    if v in tabuleiro[:, j]:
        return False
    # Verifica bloco
    b = int(n ** 0.5)
    bi, bj = i - i % b, j - j % b
    bloco = tabuleiro[bi:bi+b, bj:bj+b].flatten()
    return v not in bloco

# ---------- Gera dados de treino ----------
def gerar_dados_treino(tabuleiro, amostras=3000):
    n = tabuleiro.shape[0]
    X = []
    Y = []
    for _ in range(amostras):
        i = random.randint(0, n-1)
        j = random.randint(0, n-1)
        v = random.randint(1, n)
        if tabuleiro[i][j] != 0: # Ignora células já preenchidas
            continue
        x = torch.tensor([float(i), float(j), float(v)], dtype=torch.float32) # Converte para float
        y = 1.0 if jogada_valida(tabuleiro, i, j, v) else 0.0
        X.append(x)
        Y.append(torch.tensor([y]))
    return torch.stack(X), torch.stack(Y)


# ---------- Treina a MLP ----------
def treinar_modelo(tabuleiro):
    X, Y = gerar_dados_treino(tabuleiro)
    const_X = ltn.Variable("x", X)
    optimizer = torch.optim.Adam(model_q2.parameters(), lr=0.01) # Usa model_q2

    print("Iniciando treinamento da rede para prever jogadas válidas...")
    for epoch in range(500):
        optimizer.zero_grad()
        predicoes = ValidMove(const_X)
        loss = nn.BCELoss()(predicoes.value, Y.squeeze())
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0: # Ajuste para menos prints
            print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
    print("Treinamento concluído.")

# ---------- Sugestões por posição (válidas e inválidas) ----------
def sugerir_por_posicao_completa(tabuleiro):
    n = tabuleiro.shape[0]
    for i in range(n):
        for j in range(n):
            if tabuleiro[i][j] != 0:
                continue
            print(f"\nPosição vazia ({i}, {j}):")
            jogadas = []
            for v in range(1, n+1):
                entrada = torch.tensor([[float(i), float(j), float(v)]], dtype=torch.float32) # Converte para float
                prob = ValidMove(ltn.Constant(entrada)).value.item()
                status = "✅ válida" if prob >= 0.5 else "❌ inválida"
                jogadas.append((v, prob, status))
            # ordena do maior pro menor
            jogadas.sort(key=lambda x: -x[1])
            for v, prob, status in jogadas:
                print(f"   - Valor {v}: {status} com {prob:.4f}")

# ---------- Prever dois movimentos ----------
def prever_dois_movimentos(tabuleiro):
    n = tabuleiro.shape[0]
    resultados = []
    vazias = [(i, j) for i in range(n) for j in range(n) if tabuleiro[i][j] == 0]

    # Para evitar muitas combinações, vamos limitar as buscas ou fazer uma amostragem
    # Aqui, para fins de exemplo, vamos testar apenas um subconjunto de pares
    # Você pode ajustar isso para ser mais ou menos exaustivo
    
    # Limita o número de células vazias a considerar para o primeiro movimento
    # Isso evita uma explosão combinatória em tabuleiros muito vazios
    celulas_para_tentar = random.sample(vazias, min(len(vazias), 10)) # Testa até 10 células aleatórias para o 1º movimento

    for i1, j1 in celulas_para_tentar:
        for v1 in range(1, n+1):
            entrada1 = torch.tensor([[float(i1), float(j1), float(v1)]], dtype=torch.float32)
            prob1 = ValidMove(ltn.Constant(entrada1)).value.item()
            if prob1 < 0.7 or not jogada_valida(tabuleiro, i1, j1, v1):
                continue

            novo_tabuleiro = tabuleiro.clone()
            novo_tabuleiro[i1][j1] = v1

            # Limita o número de células vazias para o segundo movimento após o primeiro
            outras_vazias = [(i, j) for i, j in vazias if (i, j) != (i1, j1)]
            celulas_para_tentar_2 = random.sample(outras_vazias, min(len(outras_vazias), 5)) # Testa até 5 células para o 2º movimento

            for i2, j2 in celulas_para_tentar_2:
                for v2 in range(1, n+1):
                    entrada2 = torch.tensor([[float(i2), float(j2), float(v2)]], dtype=torch.float32)
                    prob2 = ValidMove(ltn.Constant(entrada2)).value.item()
                    if prob2 >= 0.7 and jogada_valida(novo_tabuleiro, i2, j2, v2):
                        resultados.append(((i1,j1,v1,prob1), (i2,j2,v2,prob2)))
    return resultados


# --- Questão 3: Heurísticas e SAT Solver para Sudoku ---

# === Parte 1: Heurística ===
# (carregar_tabuleiro_csv_q2_q3 já faz o papel de carregar_tabuleiro_csv aqui)

def analisar_opcoes_celulas(tabuleiro):
    n = tabuleiro.shape[0]
    celulas_vazias_com_opcoes = []

    for i in range(n):
        for j in range(n):
            if tabuleiro[i, j] == 0:
                opcoes_validas = []
                for v in range(1, n + 1):
                    if jogada_valida(tabuleiro, i, j, v): # Reutiliza jogada_valida da Q2
                        opcoes_validas.append(v)
                celulas_vazias_com_opcoes.append(((i, j), len(opcoes_validas)))

    celulas_vazias_com_opcoes.sort(key=lambda x: x[1])
    return celulas_vazias_com_opcoes

def recomendar_heuristica(tabuleiro):
    n = tabuleiro.shape[0]
    celulas_info = analisar_opcoes_celulas(tabuleiro)

    if not celulas_info:
        print("O tabuleiro está completo. Não há jogadas a serem feitas.")
        return

    num_celulas_vazias = len(celulas_info)

    # Contar quantas células têm poucas opções
    cont_poucas_opcoes = 0
    for _, num_opcoes in celulas_info:
        if num_opcoes <= 2: # Consideramos "poucas opções" se forem 1 ou 2
            cont_poucas_opcoes += 1

    percentual_poucas_opcoes = (cont_poucas_opcoes / num_celulas_vazias) * 100

    print(f"Análise do Tabuleiro ({n}x{n}):")
    print(f"Total de células vazias: {num_celulas_vazias}")
    print(f"Células vazias com 1 ou 2 opções: {cont_poucas_opcoes} ({percentual_poucas_opcoes:.2f}%)")

    print("\nDetalhes das células com menos opções:")
    # Imprime as 5 células com menos opções para dar uma ideia
    for k, ((r, c), num_ops) in enumerate(celulas_info[:min(5, num_celulas_vazias)]):
        print(f"  - Posição ({r},{c}): {num_ops} opções.")

    print("\n--- Recomendação de Heurística ---")

    if percentual_poucas_opcoes >= 50:
        print("Heurística Recomendada: MRV (Minimum Remaining Values)")
        print("Priorize preencher as células que possuem o menor número de opções válidas.")
        print("Isso ajuda a identificar 'gargalos' e a reduzir o espaço de busca mais rapidamente.")
        if cont_poucas_opcoes > 0 and celulas_info[0][1] == 1:
            print("Existem células com apenas 1 opção, o que indica um 'número escondido único' ou 'célula única'.")
    elif percentual_poucas_opcoes > 10:
        print("Heurística Principal: MRV (Minimum Remaining Values)")
        print("Ainda é recomendado focar nas células com menos opções, mas o tabuleiro pode ter mais flexibilidade.")
        print("Pode-se considerar a LCV (Least Constraining Value) ao escolher o valor para essas células.")
    else:
        print("Heurística Geral: MRV (Minimum Remaining Values) é sempre um bom ponto de partida.")
        print("No entanto, como muitas células têm múltiplas opções, o tabuleiro pode estar em um estágio inicial ou ter várias soluções possíveis.")
        print("Ao escolher o valor, a heurística LCV pode ser útil: escolha o valor que menos restringe as opções das células vizinhas.")

    print("\nLembre-se: Para resolver o Sudoku completamente, geralmente combinamos MRV (para escolher a célula) com LCV (para escolher o valor).")

# === Parte 2: SAT Solver ===
def var(i, j, v, n):
    return i * n * n + j * n + v

def sudoku_to_cnf(tabuleiro):
    n = tabuleiro.shape[0]
    cnf = CNF()

    # Regra 1: cada célula deve ter um valor
    for i in range(n):
        for j in range(n):
            cnf.append([var(i, j, v, n) for v in range(1, n + 1)])

    # Regra 2: cada célula pode ter no máximo um valor
    for i in range(n):
        for j in range(n):
            for v1 in range(1, n):
                for v2 in range(v1 + 1, n + 1):
                    cnf.append([-var(i, j, v1, n), -var(i, j, v2, n)])

    # Regra 3: linha
    for i in range(n):
        for v in range(1, n + 1):
            for j1 in range(n):
                for j2 in range(j1 + 1, n):
                    cnf.append([-var(i, j1, v, n), -var(i, j2, v, n)])

    # Regra 4: coluna
    for j in range(n):
        for v in range(1, n + 1):
            for i1 in range(n):
                for i2 in range(i1 + 1, n):
                    cnf.append([-var(i1, j, v, n), -var(i2, j, v, n)])

    # Regra 5: bloco
    b = int(math.sqrt(n))
    for bi in range(0, n, b):
        for bj in range(0, n, b):
            for v in range(1, n + 1):
                cells = []
                for i in range(bi, bi + b):
                    for j in range(bj, bj + b):
                        cells.append((i, j))
                for idx1 in range(len(cells)):
                    for idx2 in range(idx1 + 1, len(cells)):
                        i1, j1 = cells[idx1]
                        i2, j2 = cells[idx2]
                        cnf.append([-var(i1, j1, v, n), -var(i2, j2, v, n)])

    # Regra 6: células preenchidas
    for i in range(n):
        for j in range(n):
            v = int(tabuleiro[i, j].item())
            if v != 0:
                cnf.append([var(i, j, v, n)])

    return cnf

def resolver_sudoku_sat(tabuleiro):
    cnf = sudoku_to_cnf(tabuleiro)
    solver = Glucose3()
    solver.append_formula(cnf)

    if solver.solve():
        modelo = solver.get_model()
        n = tabuleiro.shape[0]
        sol = torch.zeros_like(tabuleiro)

        for lit in modelo:
            if lit > 0:
                v = lit - 1
                i = v // (n * n)
                j = (v % (n * n)) // n
                valor = (v % n) + 1
                sol[i, j] = valor
        print("\nSolução encontrada com SAT Solver:")
        print(sol.int().numpy())
    else:
        print("\nO tabuleiro não tem solução (insatisfatível).")


# --- Execução Principal ---
if __name__ == "__main__":
    print("--- Executando Questão 1: Classificação de Sudoku Completo ---")
    caminho_q1 = "tabuleiro4x4-invalido.csv"  # Ajuste o caminho conforme a localização do seu arquivo
    tabuleiro_q1 = carregar_tabuleiro_csv_q1(caminho_q1)
    res_q1 = classificar_tabuleiro(tabuleiro_q1)
    print("Resultado da Questão 1:", "1 - Tabuleiro VÁLIDO!" if res_q1 else "0 - Tabuleiro INVÁLIDO!")
    print("\n" + "="*80 + "\n")

    print("--- Executando Questão 2: Previsão de Jogadas com LTN e MLP ---")
    caminho_q2 = "tabuleiro4x4-parcial.csv" # Ajuste o caminho conforme a localização do seu arquivo
    tabuleiro_q2 = carregar_tabuleiro_csv_q2_q3(caminho_q2)

    treinar_modelo(tabuleiro_q2)

    print("\nAnálise completa por posição (válidas e inválidas):")
    sugerir_por_posicao_completa(tabuleiro_q2)

    print("\nSimulando dois movimentos com alta probabilidade...")
    resultados_q2 = prever_dois_movimentos(tabuleiro_q2)
    for (i1, j1, v1, p1), (i2, j2, v2, p2) in resultados_q2:
        print(f"1º movimento ({i1},{j1})={v1} ({p1:.2f}) → 2º movimento ({i2},{j2})={v2} ({p2:.2f})")
    print("\n" + "="*80 + "\n")

    print("--- Executando Questão 3: Heurísticas e SAT Solver ---")
    caminho_q3 = "tabuleiro4x4-vazio.csv"  # Ajuste o caminho conforme a localização do seu arquivo
    tabuleiro_q3 = carregar_tabuleiro_csv_q2_q3(caminho_q3)

    print("Análise heurística:")
    recomendar_heuristica(tabuleiro_q3)

    print("\nTentando resolver com SAT solver:")
    resolver_sudoku_sat(tabuleiro_q3)

    print("\nConsiderações sobre o uso de LTN:")
    print("""
    Sim, seria possível resolver o Sudoku utilizando LTN (Logic Tensor Networks).
    A LTN permite integrar lógica simbólica (como as regras do Sudoku) com aprendizado baseado em tensores, possibilitando que as restrições sejam tratadas como fórmulas lógicas com graus de verdade (fuzzy logic).

    Nesse contexto, o Sudoku pode ser formulado com predicados como:
    - cell(i, j, v): verdadeiro se a célula (i, j) contém o valor v;
    - diff(A, B): verdadeiro se os valores A e B são diferentes (usado para garantir valores únicos em linhas, colunas e blocos).

    As regras tradicionais do Sudoku são inseridas como axiomas lógicos no sistema LTN, e a rede é treinada para satisfazê-las com a maior verdade possível.

    Isso é útil especialmente em casos com ruído ou tabuleiros incompletos, onde uma solução aproximada ainda é aceitável.
    Além disso, a LTN pode combinar aprendizado supervisionado (usando exemplos de Sudokus resolvidos) com raciocínio lógico simbólico, o que é vantajoso em cenários com poucos dados mas muitas regras explícitas.
    """)