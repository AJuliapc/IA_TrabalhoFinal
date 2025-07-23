import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import ltn
import sys
import os
import copy # Para copy.deepcopy

# --- Importante: Antes de rodar este script, instale as bibliotecas: ---
# pip install numpy pandas torch ltn pysat
# pip install git+https://github.com/tommasocarraro/LTNtorch
# Se ainda tiver problemas com LTNtorch, tente:
# $env:PYTHONIOENCODING="utf-8"
# pip install git+https://github.com/tommasocarraro/LTNtorch
# ----------------------------------------------------------------------

# --- Constantes Globais (serão definidas dinamicamente por set_board_size) ---
BOARD_SIZE = None
BLOCK_SIZE = None

# --- Predicado LTN "Different" ---
class DifferentModel(nn.Module):
    def forward(self, x, y):
        # Garante que x e y são tensores e têm a mesma forma
        x_tensor = x.value if isinstance(x, ltn.Constant) else x
        y_tensor = y.value if isinstance(y, ltn.Constant) else y
        return 1.0 - torch.exp(-100 * torch.abs(x_tensor - y_tensor))

Different = ltn.Predicate(DifferentModel())

# --- Funções Auxiliares Comuns ---
def set_board_size(board_np):
    global BOARD_SIZE, BLOCK_SIZE
    BOARD_SIZE = board_np.shape[0]
    BLOCK_SIZE = int(np.sqrt(BOARD_SIZE))
    if BLOCK_SIZE * BLOCK_SIZE != BOARD_SIZE:
        print(f"Erro: O tamanho do tabuleiro {BOARD_SIZE} não é um quadrado perfeito (necessário para Sudoku).")
        return False
    return True

def carregar_tabuleiro_csv(path):
    print(f"\nLendo o tabuleiro do arquivo '{path}'...")
    try:
        df = pd.read_csv(path, header=None)
        board_np = df.values.astype(int)

        if not set_board_size(board_np):
            return None, None

        board_tensor = torch.tensor(board_np, dtype=torch.float32)

        if board_np.shape[0] != board_np.shape[1]:
            print(f"ERRO: O tabuleiro no arquivo '{path}' tem dimensões não quadradas {board_np.shape}.")
            return None, None

        # Agora que BOARD_SIZE está definido, podemos validar os valores
        if np.any(board_np < 0) or np.any(board_np > BOARD_SIZE):
            print(f"ERRO: O tabuleiro contém números fora do intervalo válido [0, {BOARD_SIZE}].")
            return None, None

        print("Tabuleiro Lido:\n", board_np)
        return board_np, board_tensor

    except FileNotFoundError:
        print(f"ERRO: Arquivo '{path}' não encontrado.")
        return None, None
    except ValueError:
        print(f"ERRO: O arquivo '{path}' não parece ser um CSV válido para o tabuleiro (contém não-números ou formato incorreto).")
        return None, None
    except Exception as e:
        print(f"ERRO inesperado ao carregar o tabuleiro: {e}")
        return None, None

def get_empty_cells(board_np):
    cells = []
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board_np[r, c] == 0:
                cells.append((r, c))
    return cells

def violates_constraint(board_np, move):
    row, col, digit = move

    if not (1 <= digit <= BOARD_SIZE):
        return True # Dígito fora do intervalo permitido

    # Verifica linha
    if digit in board_np[row, :]:
        return True
    # Verifica coluna
    if digit in board_np[:, col]:
        return True
    # Verifica bloco
    start_row, start_col = (row // BLOCK_SIZE) * BLOCK_SIZE, (col // BLOCK_SIZE) * BLOCK_SIZE
    if digit in board_np[start_row:start_row + BLOCK_SIZE, start_col:start_col + BLOCK_SIZE]:
        return True
    return False

# --- Lógica da Questão 1: Classificar Tabuleiro Fechado (usando LTN Different) ---
def gerar_axiomas_fechado(tabuleiro_tensor):
    n = tabuleiro_tensor.shape[0]
    b = int(n ** 0.5)
    axiomas = []

    # Verificação de linhas
    for i in range(n):
        filled_values_in_row = []
        for j in range(n):
            val = tabuleiro_tensor[i, j].item()
            if val != 0: # Apenas valores preenchidos importam
                filled_values_in_row.append(ltn.Constant(torch.tensor([val])))

        for k1 in range(len(filled_values_in_row)):
            for k2 in range(k1 + 1, len(filled_values_in_row)):
                axiomas.append(Different(filled_values_in_row[k1], filled_values_in_row[k2]))

    # Verificação de colunas
    for j in range(n):
        filled_values_in_col = []
        for i in range(n):
            val = tabuleiro_tensor[i, j].item()
            if val != 0:
                filled_values_in_col.append(ltn.Constant(torch.tensor([val])))

        for k1 in range(len(filled_values_in_col)):
            for k2 in range(k1 + 1, len(filled_values_in_col)):
                axiomas.append(Different(filled_values_in_col[k1], filled_values_in_col[k2]))

    # Verificação de blocos
    for bi in range(0, n, b):
        for bj in range(0, n, b):
            filled_values_in_block = []
            for i in range(bi, bi + b):
                for j in range(bj, bj + b):
                    val = tabuleiro_tensor[i, j].item()
                    if val != 0:
                        filled_values_in_block.append(ltn.Constant(torch.tensor([val])))

            for k1 in range(len(filled_values_in_block)):
                for k2 in range(k1 + 1, len(filled_values_in_block)):
                    axiomas.append(Different(filled_values_in_block[k1], filled_values_in_block[k2]))

    return axiomas

def classificar_cenario1_tabuleiro_fechado(tabuleiro_np, tabuleiro_tensor, limiar=0.99):
    print("\n--- CENÁRIO 1: CLASSIFICAR TABULEIRO FECHADO (Questão 1) ---")

    if np.any(tabuleiro_np == 0):
        print("Aviso: O tabuleiro contém células vazias (0). A classificação verificará apenas a validade dos números preenchidos.")
        print("Para tabuleiros abertos, as Questões 2 e 3 são mais apropriadas.")

    axiomas = gerar_axiomas_fechado(tabuleiro_tensor)

    if not axiomas:
        score = torch.tensor(1.0) # Se não há axiomas (ex: tabuleiro vazio), considera válido
    else:
        verdades = torch.stack([ax.value for ax in axiomas]).squeeze()
        score = verdades.min()

    print(f"Satisfação geral dos axiomas LTN: {score.item():.4f}")

    if score.item() >= limiar:
        print(f"Classificação para CENÁRIO 1: 1 - ✅ Tabuleiro VÁLIDO (Satisfação >= {limiar})!")
        return 1
    else:
        print(f"Classificação para CENÁRIO 1: 0 - ❌ Tabuleiro INVÁLIDO (Satisfação < {limiar})!")
        return 0

# --- Lógica da Questão 2: Classificar Tabuleiro Aberto e Cenários (usando Numpy) ---
def check_sem_solucao(board_np):
    empty_cells = get_empty_cells(board_np)
    if not empty_cells: # Se não há células vazias, não pode estar "sem solução" por falta de espaço
        return False

    # Verifica se existe pelo menos uma célula vazia onde *todos* os dígitos de 1 a BOARD_SIZE
    # resultariam em uma violação. Se houver tal célula, o tabuleiro está "sem solução".
    for r_empty, c_empty in empty_cells:
        all_digits_violate = True
        for digit_to_try in range(1, BOARD_SIZE + 1):
            temp_move = (r_empty, c_empty, digit_to_try)
            if not violates_constraint(board_np, temp_move):
                all_digits_violate = False
                break
        if all_digits_violate:
            # print(f"DIAGNÓSTICO: Célula ({r_empty},{c_empty}) não aceita nenhum dígito de 1 a {BOARD_SIZE}.")
            return True # Encontrou uma célula que não pode ser preenchida

    return False # Se todas as células vazias têm pelo menos uma opção, não está "sem solução" por enquanto

def evaluate_one_move(board_np):
    empty_cells = get_empty_cells(board_np)
    moves_leading_to_sem_solucao = []
    moves_maintaining_solucao_possivel = []

    for row, col in empty_cells:
        for digit in range(1, BOARD_SIZE + 1):
            move = (row, col, digit)

            if violates_constraint(board_np, move):
                continue # O movimento atual já viola as regras, não é uma jogada válida

            copied_board = np.copy(board_np)
            copied_board[row, col] = digit

            # Após o movimento, verifica se o novo estado leva a um impasse imediato
            if check_sem_solucao(copied_board):
                moves_leading_to_sem_solucao.append(move)
            else:
                moves_maintaining_solucao_possivel.append(move)
    return moves_leading_to_sem_solucao, moves_maintaining_solucao_possivel

def evaluate_two_moves(board_np):
    empty_cells = get_empty_cells(board_np)
    two_moves_leading_to_sem_solucao = []
    two_moves_maintaining_solucao_possivel = []

    # Limita o número de células para testar para o primeiro movimento para evitar explosão combinatória
    # Em um tabuleiro 9x9, isso pode ser *muito* lento sem otimizações ou limites.
    # Exemplo: random.sample(empty_cells, min(len(empty_cells), 5))
    
    # Para tabuleiros grandes, considere usar um limite de iterações ou amostragem
    # Ou implementar um algoritmo de busca com poda (backtracking) para verificar "solvabilidade real"

    # Itera sobre as células vazias para o primeiro movimento
    for r1, c1 in empty_cells:
        for d1 in range(1, BOARD_SIZE + 1):
            first_move = (r1, c1, d1)

            if violates_constraint(board_np, first_move):
                continue

            board_after_first_move = copy.deepcopy(board_np)
            board_after_first_move[r1, c1] = d1

            # Verifica se o primeiro movimento já leva a um estado sem solução
            if check_sem_solucao(board_after_first_move):
                two_moves_leading_to_sem_solucao.append((first_move, "Imediato - 1º passo leva a Sem Solução"))
                continue

            empty_cells_after_first_move = get_empty_cells(board_after_first_move)

            # Se o tabuleiro ficou completo após o primeiro movimento, não há segundo movimento
            if not empty_cells_after_first_move:
                two_moves_maintaining_solucao_possivel.append((first_move, "Tabuleiro Completo"))
                continue

            # Itera sobre as células vazias (após o primeiro movimento) para o segundo movimento
            for r2, c2 in empty_cells_after_first_move:
                # Evita que o segundo movimento seja na mesma célula que acabou de ser preenchida
                if (r1, c1) == (r2, c2):
                    continue

                for d2 in range(1, BOARD_SIZE + 1):
                    second_move = (r2, c2, d2)

                    if violates_constraint(board_after_first_move, second_move):
                        continue

                    board_after_two_moves = copy.deepcopy(board_after_first_move)
                    board_after_two_moves[r2, c2] = d2

                    # Verifica se o tabuleiro está sem solução após o segundo movimento
                    if check_sem_solucao(board_after_two_moves):
                        two_moves_leading_to_sem_solucao.append((first_move, second_move))
                    else:
                        two_moves_maintaining_solucao_possivel.append((first_move, second_move))

    return two_moves_leading_to_sem_solucao, two_moves_maintaining_solucao_possivel

def classificar_cenario2_tabuleiro_aberto(board_np):
    print("\n--- CENÁRIO 2: CLASSIFICAR TABULEIRO ABERTO (Questão 2) ---")

    empty_cells = get_empty_cells(board_np)
    if not empty_cells:
        print("O tabuleiro está completo. Não há células vazias para analisar.")
        print("Classificação para CENÁRIO 2: O tabuleiro está preenchido, não se aplica 'aberto'.")
        return

    is_sem_solucao = check_sem_solucao(board_np)

    if is_sem_solucao:
        print("Classificação para CENÁRIO 2: 1) Sem Solução")
    else:
        print("Classificação para CENÁRIO 2: 2) Solução Possível")

        print("\n--- CENÁRIO 2a: AVALIAÇÃO DE MOVIMENTOS EM UM PASSO ---")
        moves_leading_to_sem_solucao_one_step, moves_maintaining_solucao_possivel_one_step = evaluate_one_move(board_np)
        print(f"Movimentos que levam a 'Sem Solução': {moves_leading_to_sem_solucao_one_step}")
        print(f"Movimentos que mantêm 'Solução Possível': {moves_maintaining_solucao_possivel_one_step}")

        if moves_leading_to_sem_solucao_one_step:
            digits_causing_impasse = {}
            for move in moves_leading_to_sem_solucao_one_step:
                r, c, d = move
                digits_causing_impasse[d] = digits_causing_impasse.get(d, 0) + 1
            sorted_digits_impasse = sorted(digits_causing_impasse.items(), key=lambda item: item[1], reverse=True)
            print(f"Numerais com maior probabilidade de levar a 'Sem Solução' (em 1 movimento): {sorted_digits_impasse}")
        else:
            print("Nenhum movimento em um passo leva a 'Sem Solução'.")

        print("\n--- CENÁRIO 2b: AVALIAÇÃO DE SEQUÊNCIAS DE DOIS MOVIMENTOS ---")
        two_moves_leading_to_sem_solucao, two_moves_maintaining_solucao_possivel = evaluate_two_moves(board_np)
        print(f"Sequências de dois movimentos que levam a 'Sem Solução': {two_moves_leading_to_sem_solucao}")
        print(f"Sequências de dois movimentos que mantêm 'Solução Possível': {two_moves_maintaining_solucao_possivel}")

        if two_moves_leading_to_sem_solucao:
            digits_causing_impasse_two_steps = {}
            for seq in two_moves_leading_to_sem_solucao:
                # A seq pode ser ((r1,c1,d1), (r2,c2,d2)) ou ((r1,c1,d1), "Imediato...")
                if isinstance(seq[1], tuple): # É um par de movimentos
                    move1_digit = seq[0][2]
                else: # É um movimento imediato
                    move1_digit = seq[0][2]
                digits_causing_impasse_two_steps[move1_digit] = digits_causing_impasse_two_steps.get(move1_digit, 0) + 1
            sorted_digits_impasse_two_steps = sorted(digits_causing_impasse_two_steps.items(), key=lambda item: item[1], reverse=True)
            print(f"Numerais com maior probabilidade de levar a 'Sem Solução' (em 2 movimentos, considerando o 1º dígito): {sorted_digits_impasse_two_steps}")
        else:
            print("Nenhuma sequência de dois movimentos leva a 'Sem Solução'.")


# --- Lógica da Questão 3: Indicar Heurísticas Mais Recomendadas (usando Numpy) ---
def initialize_binary_variables(board_np):
    binary_vars = np.zeros((BOARD_SIZE, BOARD_SIZE, BOARD_SIZE + 1), dtype=int)
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            num = board_np[i, j]
            if num != 0:
                if 1 <= num <= BOARD_SIZE:
                    binary_vars[i, j, num] = 1
    return binary_vars

def is_valid_for_heuristic(binary_vars_3d, row, col, num):
    # Verifica se o número 'num' já está presente na linha 'row'
    if np.sum(binary_vars_3d[row, :, num]) > 0:
        return False
    # Verifica se o número 'num' já está presente na coluna 'col'
    if np.sum(binary_vars_3d[:, col, num]) > 0:
        return False
    # Verifica se o número 'num' já está presente no bloco
    block_row_start = (row // BLOCK_SIZE) * BLOCK_SIZE
    block_col_start = (col // BLOCK_SIZE) * BLOCK_SIZE
    if np.sum(binary_vars_3d[block_row_start:block_row_start + BLOCK_SIZE,
                             block_col_start:block_col_start + BLOCK_SIZE, num]) > 0:
        return False
    return True

def heuristic_mrv(board_np_original):
    binary_vars = initialize_binary_variables(board_np_original)
    empty_cells = get_empty_cells(board_np_original)

    cell_options = []
    for r, c in empty_cells:
        options = []
        for num in range(1, BOARD_SIZE + 1):
            # Se o número 'num' pode ser colocado em (r, c) sem violar as regras atuais
            # (considerando os números já preenchidos no tabuleiro original)
            if not violates_constraint(board_np_original, (r, c, num)):
                options.append(num)
        cell_options.append(((r, c), options))

    cell_options.sort(key=lambda x: len(x[1]))
    return cell_options

def heuristic_most_constrained_digit(board_np_original):
    # Usamos o tabuleiro original para ver onde CADA DÍGITO pode ser colocado
    empty_cells = get_empty_cells(board_np_original)

    digit_spaces = {}
    for num in range(1, BOARD_SIZE + 1):
        count = 0
        for r, c in empty_cells:
            # Verifica quantas posições vazias o 'num' pode ser colocado
            if not violates_constraint(board_np_original, (r, c, num)):
                count += 1
        digit_spaces[num] = count

    # Ordena os dígitos pelo número de espaços disponíveis (os mais restritos vêm primeiro)
    sorted_digits = sorted(digit_spaces.items(), key=lambda x: x[1])
    return sorted_digits

def classificar_cenario3_indicar_heuristicas(board_np):
    print("\n--- CENÁRIO 3: INDICAR HEURÍSTICAS MAIS RECOMENDADAS (Questão 3) ---")

    empty_cells = get_empty_cells(board_np)
    if not empty_cells:
        print("O tabuleiro está completo. Não há células vazias para aplicar heurísticas.")
        return

    print("\n--- Heurística MRV (Minimum Remaining Values) ---")
    mrv_results = heuristic_mrv(board_np)
    if mrv_results:
        print("Células com o menor número de opções válidas (mais restritas):")
        for (r, c), opts in mrv_results[:min(5, len(mrv_results))]:
            print(f"   Célula ({r},{c}) → {len(opts)} possibilidade(s): {opts}")
    else:
        print("Não há células vazias para aplicar MRV.")

    print("\n--- Heurística 'Dígito Mais Restrito' ---")
    most_constrained_digit_results = heuristic_most_constrained_digit(board_np)
    if most_constrained_digit_results:
        print("Dígitos com o menor número de posições válidas no tabuleiro (mais restritos):")
        for digit, count in most_constrained_digit_results[:min(BOARD_SIZE, len(most_constrained_digit_results))]: # Limita para BOARD_SIZE dígitos
            print(f"   Dígito {digit} → {count} posição(ões) possível(is)")
    else:
        print("Não há dígitos para analisar em células vazias.")

    print("\nConsiderações sobre as heurísticas:")
    print("A heurística MRV foca na célula mais difícil de preencher. Preenchendo-a primeiro, você pode expor um impasse cedo.")
    print("A heurística do Dígito Mais Restrito foca no dígito mais difícil de alocar. Alocá-lo pode evitar problemas futuros.")
    print("Ambas são úteis para guiar a busca em problemas de satisfação de restrições.")

    print("\n--- Pergunta Final: Resolução de Sudoku com LTN (Restrições + Heurísticas) ---")
    print("Sim, é possível resolver o Sudoku usando apenas LTN combinando restrições e heurísticas.")
    print("Para isso, o modelo 'DummyModel' precisaria ser substituído por uma Rede Neural real (ex: MLP).")
    print("Essa rede aprenderia a 'confiança' de cada número em cada célula.")
    print("Os axiomas LTN (como 'Different', e outros para 'HasNumber' e 'IsFilled' para células vazias)")
    print("formariam a função de perda. Otimizadores (SGD, Adam) ajustariam os pesos da rede para maximizar")
    print("a satisfação geral dos axiomas. As heurísticas seriam incorporadas como cláusulas LTN adicionais")
    print("para guiar o aprendizado da rede, priorizando certos tipos de atribuições.")
    print("Por exemplo, penalizando soluções onde uma célula tem muitas opções ou favorecendo a alocação de dígitos raros.")
    print("Após o treinamento, o tabuleiro seria resolvido selecionando os números com maior confiança pela rede.")

# --- Função Principal (Main) para Execução ---
if __name__ == "__main__":
    # --- DEFINA A PASTA RAIZ DO SEU TRABALHO AQUI ---
    # AJUSTE ESTE CAMINHO PARA A LOCALIZAÇÃO DA SUA PASTA 'trabalho-final-IA'
    # Exemplo para Windows (se a pasta 'trabalho-final-IA' estiver em Documentos\Faculdade\IA):
    # base_directory_path = "C:/Users/anaju/OneDrive/Documentos/Faculdade/IA/trabalho-final-IA"
    # Exemplo para Linux/WSL:
    # base_directory_path = "/mnt/c/Users/anaju/OneDrive/Documentos/Faculdade/IA/trabalho-final-IA"
    # Ou se você colocou os tabuleiros diretamente na pasta do script:
    base_directory_path = os.path.dirname(os.path.abspath(__file__)) # Pega o diretório do script atual

    # Define as subpastas para os tabuleiros (ajuste se seus CSVs estão diretamente na pasta do script)
    # Se os CSVs estiverem na mesma pasta do script, você pode remover os 'os.path.join'
    tabuleiros_q1_path = os.path.join(base_directory_path, "tabuleiros-questao1")
    tabuleiros_q2_path = os.path.join(base_directory_path, "tabuleiros-questao2")
    tabuleiros_q3_path = os.path.join(base_directory_path, "tabuleiros-questao3")

    # Garante que os caminhos são tratados corretamente para diferentes SOs
    tabuleiros_q1_path = os.path.normpath(tabuleiros_q1_path)
    tabuleiros_q2_path = os.path.normpath(tabuleiros_q2_path)
    tabuleiros_q3_path = os.path.normpath(tabuleiros_q3_path)

    print(f"Caminho base detectado: {base_directory_path}")

    # Processa tabuleiros da Questão 1
    print("\n" + "="*80)
    print("==== Processando tabuleiros para CENÁRIO 1 (Questão 1) ====")
    print("="*80)
    if os.path.isdir(tabuleiros_q1_path):
        csv_files_q1 = [f for f in os.listdir(tabuleiros_q1_path) if f.endswith(".csv")]
        if not csv_files_q1:
            print(f"Nenhum arquivo CSV encontrado em '{tabuleiros_q1_path}'")
        for filename in csv_files_q1:
            file_path = os.path.join(tabuleiros_q1_path, filename)
            board_np, board_tensor = carregar_tabuleiro_csv(file_path)
            if board_np is not None and board_tensor is not None:
                print(f"\n==== ANÁLISE PARA O ARQUIVO: {filename} ====")
                classificar_cenario1_tabuleiro_fechado(board_np, board_tensor)
            else:
                print(f"\n==== Pulando arquivo: {filename} devido a erros de carregamento/validação ====")
        print("\n")
    else:
        print(f"ERRO: Pasta '{tabuleiros_q1_path}' não encontrada. Pulando Questão 1.")
        print("Certifique-se de que os arquivos CSV da Questão 1 estão na subpasta 'tabuleiros-questao1'.")

    # Processa tabuleiros das Questões 2 e 3
    print("\n" + "="*80)
    print("==== Processando tabuleiros para CENÁRIO 2 e 3 (Questões 2 e 3) ====")
    print("="*80)
    # Lista de pastas a serem processadas para as Questões 2 e 3
    q2_q3_folders = [tabuleiros_q2_path, tabuleiros_q3_path]

    for folder_path in q2_q3_folders:
        if os.path.isdir(folder_path):
            print(f"\n--- Processando tabuleiros da pasta: {os.path.basename(folder_path)} ---")
            csv_files_q2_q3 = [f for f in os.listdir(folder_path) if f.endswith(".csv")]
            if not csv_files_q2_q3:
                print(f"Nenhum arquivo CSV encontrado em '{folder_path}'")
            for filename in csv_files_q2_q3:
                file_path = os.path.join(folder_path, filename)
                board_np, board_tensor = carregar_tabuleiro_csv(file_path)
                if board_np is not None and board_tensor is not None:
                    print(f"\n==== ANÁLISE PARA O ARQUIVO: {filename} ====")
                    if np.any(board_np == 0): # Só executa Q2 e Q3 se o tabuleiro tiver células vazias
                        classificar_cenario2_tabuleiro_aberto(board_np)
                        classificar_cenario3_indicar_heuristicas(board_np)
                    else:
                        print("\n--- CENÁRIO 2 & 3: Não aplicável. O tabuleiro está completo. ---")
                        print("Este tabuleiro será classificado apenas pela Questão 1.")
                else:
                    print(f"\n==== Pulando arquivo: {filename} devido a erros de carregamento/validação ====")
        else:
            print(f"ERRO: Pasta '{os.path.basename(folder_path)}' não encontrada. Pulando processamento desta pasta.")
            print(f"Certifique-se de que os arquivos CSV para as Questões 2 e 3 estão nas subpastas apropriadas.")