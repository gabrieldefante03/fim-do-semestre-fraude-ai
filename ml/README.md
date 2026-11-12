# ML — Detecção de fraude

Modelo Random Forest treinado com o dataset [PaySim](https://www.kaggle.com/datasets/ealaxi/paysim1).
O `predict.py` é chamado pelo terminal (e pelo Java, via `ProcessBuilder`) e imprime só o resultado.

```
ml/
├── data/              dataset PaySim (fora do Git, ver "Baixar o dataset")
├── models/
│   └── modelo.pkl     modelo treinado (vai para o Git, 0,5 MB)
├── kdd_fraude.ipynb   processo KDD completo: análise, comparação de modelos e justificativa
├── train.py           treina e gera models/modelo.pkl
├── predict.py         classifica uma transação
├── requirements.txt           bibliotecas do train.py e do predict.py
└── requirements-notebook.txt  + Jupyter e matplotlib, só para o notebook
```

## Preparar o ambiente

Requer Python 3.12 ou mais novo. Dentro da pasta `ml`:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

No Linux/Mac, troque `.\.venv\Scripts\python.exe` por `.venv/bin/python` em todos os comandos.

## Predizer

O modelo já vem treinado em `models/modelo.pkl`, então não é preciso baixar o dataset para usar o `predict.py`.

```powershell
.\.venv\Scripts\python.exe predict.py --tipo TRANSFER --valor 181.00 --saldo-origem-antes 181.00 --saldo-origem-depois 0 --saldo-destino-antes 0 --saldo-destino-depois 0
```

```json
{"resultado": "FRAUDE", "probabilidade": 0.9937}
```

| Argumento | Coluna do PaySim | Descrição |
|---|---|---|
| `--tipo` | `type` | `CASH_IN`, `CASH_OUT`, `DEBIT`, `PAYMENT` ou `TRANSFER` |
| `--valor` | `amount` | valor da transação |
| `--saldo-origem-antes` | `oldbalanceOrg` | saldo de quem envia, antes |
| `--saldo-origem-depois` | `newbalanceOrig` | saldo de quem envia, depois |
| `--saldo-destino-antes` | `oldbalanceDest` | saldo de quem recebe, antes |
| `--saldo-destino-depois` | `newbalanceDest` | saldo de quem recebe, depois |

Os números usam ponto como separador decimal (`181.50`, não `181,50`). `--help` mostra a ajuda.

### Contrato com o Java

- **stdout:** sempre uma única linha JSON com `resultado` (`FRAUDE` ou `APROVADO`) e `probabilidade` de fraude (0 a 1). Nada mais é impresso no stdout.
- **stderr:** mensagens de erro. Só aparecem quando o código de saída é diferente de 0.
- **Código de saída:** `0` sucesso, `1` erro ao rodar (ex.: modelo não encontrado), `2` argumento inválido ou faltando.
- O Java deve chamar o Python do venv (`ml/.venv/Scripts/python.exe`) e passar cada argumento como um item separado do `ProcessBuilder`.
- O `predict.py` acha o modelo pelo caminho do próprio arquivo, então pode ser chamado de qualquer pasta.
- Cada chamada leva cerca de 3 segundos, quase tudo para carregar o pandas e o scikit-learn.

## Treinar de novo (opcional)

### Baixar o dataset

O arquivo tem 470 MB e por isso não vai para o Git. É preciso uma conta no Kaggle.

- **Pelo navegador:** baixe em <https://www.kaggle.com/datasets/ealaxi/paysim1>, descompacte e coloque `PS_20174392719_1491204439457_log.csv` em `ml/data/`.
- **Pelo terminal:** com o [token da API do Kaggle](https://www.kaggle.com/settings) configurado, rode na pasta `ml`:
  ```powershell
  .\.venv\Scripts\python.exe -m pip install kaggle
  .\.venv\Scripts\kaggle.exe datasets download ealaxi/paysim1 -p data --unzip
  ```

### Rodar o treino

```powershell
.\.venv\Scripts\python.exe train.py
```

Leva cerca de 30 segundos e sobrescreve `models/modelo.pkl`. O treino usa todas as fraudes e 1 milhão de transações
normais sorteadas; a avaliação usa 20% do dataset (1,27 milhão de transações) que o modelo não viu, com a proporção
real de fraudes.

## Resultado

| | Precisão | Recall |
|---|---|---|
| FRAUDE | 100,00% | 99,76% |

Das 1.643 fraudes do teste, 1.639 foram detectadas e nenhuma transação normal foi bloqueada por engano.

Esse resultado é tão alto por causa de uma característica do simulador: na fraude do PaySim o golpista transfere
exatamente o saldo inteiro da vítima (97,8% das fraudes, contra 0,0% das transações normais), e em metade das fraudes
o saldo de quem recebe não muda. Já nas transações normais o saldo de quem envia quase nunca "fecha" com o valor.
O `predict.py` calcula essas diferenças (`erroSaldoOrigem` e `erroSaldoDestino`), e são elas que o modelo mais usa.
Sem essas duas colunas, o mesmo Random Forest detecta só 78% das fraudes e bloqueia 105
transações normais por engano (ver o notebook). Em dados reais de banco a detecção não seria tão fácil.

## Notebook KDD

O `kdd_fraude.ipynb` documenta as cinco etapas do KDD (seleção, pré-processamento, transformação, mineração e
avaliação) com gráficos, compara o Random Forest com outros cinco modelos e justifica cada decisão. Ele já está salvo
executado, então dá para ler direto no GitHub. Para rodar de novo, é preciso o dataset em `ml/data/` e:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-notebook.txt
.\.venv\Scripts\python.exe -m notebook kdd_fraude.ipynb
```

Use `python.exe -m notebook` e não `jupyter.exe notebook`: se houver outro Python com Jupyter instalado no computador,
o `jupyter.exe` pode abrir esse outro, com versões diferentes das bibliotecas. No final, o notebook gera o
`models/modelo.pkl` de novo pelo `train.py`, mas só se as versões baterem com o `requirements.txt`.

## Créditos

PaySim: E. A. Lopez-Rojas, A. Elmir e S. Axelsson. *PaySim: A financial mobile money simulator for fraud detection*.
28th European Modeling and Simulation Symposium (EMSS), 2016. Licença CC BY-SA 4.0.
