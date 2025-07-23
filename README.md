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

O código foi desenvolvido para execução no **Google Colab** e em **Em Python** no VS Code ou outro meio de editar/compilar códigos, e os tabuleiros de teste são lidos a partir de arquivos `.csv`.

🔗 [Abrir no Google Colab](https://colab.research.google.com/drive/1NaGx8s9rccN70PS1UWygcNma0rPtE3oL?usp=sharing)

É disponibilizado também neste repositórioo o arquivo `.ipynb` deste Google Colab para análise dos testes feitos. Abaixo há a explicação de onde está os arquivos em **Pyhton e resumidamente o que cada script faz.**

---

### Em Python 

Este projeto também está disponível em formato `.py`, além da versão no Google Colab. Os arquivos com e sem treinamento podem ser encontrados na pasta `IAEmPython.zip`. Dentro desse `.zip`, estão incluídos:

* `script.py` (com treinamento)
* `script_sem_treinamento.py` (sem treinamento)

<img width="1682" height="386" alt="image" src="https://github.com/user-attachments/assets/3ad721ea-895b-4f9e-b406-ea4e6f4405bb" />

Frisa-se que a principal diferença entre esses códigos é que o `script_sem_treinamento.py` usa regras programadas diretamente para validar e analisar o Sudoku. Ele sabe as regras "de antemão". Já o `script.py` introduz uma rede neural que "aprende" as regras a partir de exemplos, permitindo que ela preveja a validade das jogadas com base em probabilidades. O segundo código também inclui um solver SAT para encontrar a solução exata.

Ambos os scripts estão prontos para execução, com os caminhos de teste já configurados. O pacote também inclui as pastas com os arquivos `.csv` contendo os tabuleiros de Sudoku utilizados em diferentes cenários - válido, inválido, vazio e solucionável.

Os `.csv` estão disponíveis nas pastas, desta forma:

  
        ├── script.py
        ├── script_sem_treinamento.py
        ├── tabuleiros-questao1/
        │   └── tabuleiro4x4-invalido.csv
        |   └── ...
        ├── tabuleiros-questao2/
        │   └── tabuleiro4x4-parcial.csv
        |   └── ...
        └── tabuleiros-questao3/
        |   └── tabuleiro4x4-solucionavel.csv
        |   └── ...    

E são divididos em:
* Válido: Um tabuleiro completo onde todas as regras do Sudoku (números únicos por linha, coluna e bloco) são respeitadas. Ideal para testar a validação do Cenário 1.
* Inválido: Um tabuleiro completo ou parcialmente preenchido que contém pelo menos uma violação das regras do Sudoku. Serve para verificar a capacidade do sistema em identificar erros.
* Vazio: Um tabuleiro onde todas as células estão marcadas com '0'. Este caso extremo testa o comportamento do sistema em um estado inicial sem preenchimentos.
* Parcial: Um tabuleiro incompleto (contém '0's) onde os números preenchidos até o momento não violam as regras, mas ele ainda não tem uma solução óbvia ou garantida. Usado para testar as heurísticas e a análise de movimentos.
* Solucionável: Um tabuleiro incompleto que, embora tenha células vazias, comprovadamente possui uma ou mais soluções válidas. É o tipo de tabuleiro que o solver SAT no segundo script tentaria resolver.

**Observação**: Caso queira executar um Sudoku à parte, adicione o arquivo `.csv` em uma das pastas (`tabuleiro-questao1`, `tabuleiro-questao2`, `tabuleiro-questao3`).

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

### Questão Teórica

Seria possivel resolver o Sudoko com LTN?

-  Sim, seria possível resolver o Sudoku utilizando LTN (Logic Tensor Networks).
    A LTN permite integrar lógica simbólica (como as regras do Sudoku) com aprendizado baseado em tensores, possibilitando que as restrições sejam tratadas como fórmulas lógicas com graus de verdade (fuzzy logic).

    Nesse contexto, o Sudoku pode ser formulado com predicados como:
    - cell(i, j, v): verdadeiro se a célula (i, j) contém o valor v;
    - diff(A, B): verdadeiro se os valores A e B são diferentes (usado para garantir valores únicos em linhas, colunas e blocos).

    As regras tradicionais do Sudoku são inseridas como axiomas lógicos no sistema LTN, e a rede é treinada para satisfazê-las com a maior verdade possível.

    Isso é útil especialmente em casos com ruído ou tabuleiros incompletos, onde uma solução aproximada ainda é aceitável.
    Além disso, a LTN pode combinar aprendizado supervisionado (usando exemplos de Sudokus resolvidos) com raciocínio lógico simbólico, o que é vantajoso em cenários com poucos dados mas muitas regras explícitas.

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
