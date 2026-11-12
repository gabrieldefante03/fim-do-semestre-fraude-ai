"""Classifica uma transação como FRAUDE ou APROVADO usando o modelo gerado pelo train.py.

Uso:
    python predict.py --tipo TRANSFER --valor 181.00 --saldo-origem-antes 181.00 \
        --saldo-origem-depois 0 --saldo-destino-antes 0 --saldo-destino-depois 0

Saída (stdout): uma única linha JSON, por exemplo {"resultado": "FRAUDE", "probabilidade": 0.9987}.
Qualquer erro vai para o stderr com código de saída diferente de 0, para não quebrar o parser do Java.
"""
import argparse
import json
import math
import sys
import warnings
from pathlib import Path

import joblib
import pandas as pd

CAMINHO_MODELO = Path(__file__).resolve().parent / "models" / "modelo.pkl"
LIMIAR = 0.5
TIPOS = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
COLUNAS_ENTRADA = ["type", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]


def preparar_features(df):
    """Adiciona as colunas de erro de saldo. Usada no treino e na predição, para as duas ficarem iguais."""
    df = df.copy()
    df["erroSaldoOrigem"] = df["newbalanceOrig"] + df["amount"] - df["oldbalanceOrg"]
    df["erroSaldoDestino"] = df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]
    return df


def numero_nao_negativo(texto):
    try:
        numero = float(texto)
    except ValueError:
        raise argparse.ArgumentTypeError(f"'{texto}' nao e um numero (use ponto como separador decimal)")
    if not math.isfinite(numero) or numero < 0:
        raise argparse.ArgumentTypeError(f"'{texto}' deve ser um numero maior ou igual a zero")
    return numero


def ler_argumentos():
    parser = argparse.ArgumentParser(description="Detecta fraude em uma transacao (modelo treinado no PaySim).")
    parser.add_argument("--tipo", required=True, type=str.upper, choices=TIPOS, help="tipo da transacao")
    parser.add_argument("--valor", required=True, type=numero_nao_negativo, help="valor da transacao")
    parser.add_argument("--saldo-origem-antes", required=True, type=numero_nao_negativo)
    parser.add_argument("--saldo-origem-depois", required=True, type=numero_nao_negativo)
    parser.add_argument("--saldo-destino-antes", required=True, type=numero_nao_negativo)
    parser.add_argument("--saldo-destino-depois", required=True, type=numero_nao_negativo)
    return parser.parse_args()


def main():
    # Avisos de bibliotecas (ex.: versão do scikit-learn) não podem aparecer na saída lida pelo Java
    warnings.filterwarnings("ignore")
    args = ler_argumentos()

    if not CAMINHO_MODELO.exists():
        print(f"erro: modelo nao encontrado em {CAMINHO_MODELO}. Rode o train.py primeiro.", file=sys.stderr)
        return 1

    transacao = pd.DataFrame([{
        "type": args.tipo,
        "amount": args.valor,
        "oldbalanceOrg": args.saldo_origem_antes,
        "newbalanceOrig": args.saldo_origem_depois,
        "oldbalanceDest": args.saldo_destino_antes,
        "newbalanceDest": args.saldo_destino_depois,
    }])
    modelo = joblib.load(CAMINHO_MODELO)
    probabilidade = float(modelo.predict_proba(preparar_features(transacao))[0, 1])

    resultado = "FRAUDE" if probabilidade >= LIMIAR else "APROVADO"
    print(json.dumps({"resultado": resultado, "probabilidade": round(probabilidade, 4)}))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as erro:
        print(f"erro: {erro}", file=sys.stderr)
        sys.exit(1)
