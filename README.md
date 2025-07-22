# Trabalho Final – Inteligência Artificial

## Integrantes

- Alberth Viana de Lima  
- Ana Júlia Pereira Corrêa  
- Daniel Silveira Gonzalez  
- Guilherme Sahdo Maciel  
- Júlio Melo Campos  
- Stepheson Custódio  

---

## Descrição do Projeto

Este projeto implementa um sistema de classificação de tabuleiros de Sudoku (4×4 e 9×9) utilizando Logic Tensor Networks (LTN) com a biblioteca LTNTorch. Além da validação lógica, o sistema incorpora uma Rede Neural Perceptron Multicamadas (MLP) para auxiliar na classificação de tabuleiros abertos e na sugestão de jogadas, aprendendo o conceito de movimentos válidos através de treinamento.

O código foi desenvolvido para execução no **Google Colab**, e os tabuleiros são lidos a partir de arquivos `.csv`.

🔗 [Abrir no Google Colab](https://colab.research.google.com/drive/1NaGx8s9rccN70PS1UWygcNma0rPtE3oL?usp=sharing)

---

## Objetivos

- Representar as regras do Sudoku como **axiomas lógicos** em LTN  
- Utilizar **classificadores** para analisar tabuleiros completos ou incompletos  
- Aplicar **heurísticas** para avaliar possíveis soluções e recomendar estratégias  

---

## Funcionalidades

### Questão 1 – Classificação de Tabuleiro Fechado
Verifica se um tabuleiro **completo** (sem células vazias) está correto de acordo com as regras do Sudoku.

- **Retorno:**  
  `1` → tabuleiro válido  
  `0` → tabuleiro inválido  

---

### Questão 2 – Classificação de Tabuleiro Aberto (com heurísticas)
Dado um tabuleiro com **células em branco**:

- Detecta se há numerais que não podem ser inseridos sem violar regras (sem solução)
- Caso contrário, classifica como com **solução possível**  
- Indica quais **jogadas** (1 ou 2 movimentos) mantêm o estado solucionável

---

### Questão 3 – Recomendação de Heurísticas

- Compara diferentes **conjuntos de heurísticas**  
- Gera **fórmulas lógicas** para cada conjunto  
- Executa um **solucionador lógico** (ex.: SAT-solver)  
- Avalia se é possível resolver o Sudoku apenas com **LTN + heurísticas**  

---

## Requisitos

- Python `3.8+`  
- `LTNTorch` instalado no ambiente
- Tabuleiros de entrada no formato `.csv`  

---

## Testes

Os testes foram realizados com tabuleiros armazenados em arquivos `.csv`, disponíveis nos links abaixo:

- 📁 [Google Drive – Arquivos de Teste](https://drive.google.com/drive/folders/1qoAyDbs-ToL6Z1n_wIF_OJC83J7GW6Mj?usp=sharing)  
- 📂 Também incluídos no repositório local, na pasta `Testes`  

---

## Referência

> Designing Logic Tensor Networks for Visual Sudoku Puzzle Classification
