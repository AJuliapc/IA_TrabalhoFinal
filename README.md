# Trabalho Final - Inteligência Artificial

Este projeto implementa um sistema de classificação de tabuleiros de Sudoku (4x4 e 9x9) utilizando Logic Tensor Networks (LTN) com a biblioteca LTNTorch. O código é executado diretamente no Google Colab, e os tabuleiros são lidos a partir de arquivos `.csv`.

## Objetivos

* Representar as regras do Sudoku como axiomas lógicos em LTN
* Utilizar classificadores para analisar tabuleiros completos ou incompletos
* Aplicar heurísticas para avaliar possíveis soluções e recomendar estratégias

## Funcionalidades

### Questão 1 – Classificar tabuleiro fechado

Dado um tabuleiro completo (sem células vazias), o sistema verifica se todas as regras do Sudoku são satisfeitas.
Retorna:

* 1 para tabuleiros válidos
* 0 para tabuleiros com erros

### Questão 2 – Classificar tabuleiro aberto (com heurísticas)

Dado um tabuleiro com células em branco:

* Detecta se há numerais que não podem ser inseridos sem violar regras (sem solução)
* Caso contrário, classifica como com solução possível
* Indica quais jogadas (1 ou 2 movimentos) mantêm o estado solucionável

### Questão 3 – Recomendação de heurísticas

* Compara diferentes conjuntos de heurísticas
* Gera fórmulas lógicas para cada conjunto
* Executa um solucionador lógico (como um SAT-solver)
* Avalia se é possível resolver o Sudoku apenas com LTN + heurísticas

## Requisitos

* Python 3.8+
* LTNTorch instalado no ambiente
* Tabuleiros em arquivos `.csv`

## Referência

* *Designing Logic Tensor Networks for Visual Sudoku Puzzle Classification*
