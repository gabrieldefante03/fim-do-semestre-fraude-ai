"""Treina o modelo de detecção de fraude (Random Forest) com o dataset PaySim e salva em models/modelo.pkl.

Uso:
    python train.py
"""
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from predict import CAMINHO_MODELO, COLUNAS_ENTRADA, LIMIAR, TIPOS, preparar_features

CAMINHO_DATASET = Path(__file__).resolve().parent / "data" / "PS_20174392719_1491204439457_log.csv"
# Fraudes são só 0,13% do PaySim: o treino usa todas as fraudes e uma amostra das transações normais
NAO_FRAUDES_NO_TREINO = 1_000_000
SEMENTE = 42


def carregar_dataset():
    if not CAMINHO_DATASET.exists():
        sys.exit(f"Dataset não encontrado em {CAMINHO_DATASET}\nVeja no ml/README.md como baixar o PaySim do Kaggle.")
    print(f"Lendo {CAMINHO_DATASET.name}...")
    df = pd.read_csv(CAMINHO_DATASET, usecols=[*COLUNAS_ENTRADA, "isFraud"])
    print(f"{len(df):,} transações, {df['isFraud'].sum():,} fraudes ({df['isFraud'].mean():.2%})")
    return df


def amostrar_treino(X, y):
    gerador = np.random.default_rng(SEMENTE)
    fraudes = np.flatnonzero(y.to_numpy() == 1)
    normais = np.flatnonzero(y.to_numpy() == 0)
    normais = gerador.choice(normais, size=min(NAO_FRAUDES_NO_TREINO, len(normais)), replace=False)
    indices = np.concatenate([fraudes, normais])
    return X.iloc[indices], y.iloc[indices]


def criar_modelo():
    preprocessamento = ColumnTransformer(
        [("tipo", OneHotEncoder(categories=[TIPOS], handle_unknown="ignore"), ["type"])],
        remainder="passthrough",
    )
    floresta = RandomForestClassifier(
        n_estimators=100, max_depth=15, min_samples_leaf=2, n_jobs=-1, random_state=SEMENTE
    )
    return Pipeline([("preprocessamento", preprocessamento), ("floresta", floresta)])


def avaliar(modelo, X_teste, y_teste):
    previsto = (modelo.predict_proba(X_teste)[:, 1] >= LIMIAR).astype(int)
    print(f"\nAvaliação em {len(y_teste):,} transações que o modelo não viu (limiar {LIMIAR}):")
    print(classification_report(y_teste, previsto, target_names=["APROVADO", "FRAUDE"], digits=4))
    vn, fp, fn, vp = confusion_matrix(y_teste, previsto).ravel()
    print(f"Fraudes detectadas: {vp:,} | fraudes que passaram: {fn:,}")
    print(f"Normais aprovadas:  {vn:,} | normais bloqueadas por engano: {fp:,}")


def main():
    inicio = time.time()
    df = carregar_dataset()
    y = df.pop("isFraud")
    X = preparar_features(df)

    # O teste mantém a proporção real de fraudes, para a avaliação refletir o uso de verdade
    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEMENTE
    )
    X_treino, y_treino = amostrar_treino(X_treino, y_treino)

    print(f"\nTreinando Random Forest com {len(y_treino):,} transações ({y_treino.sum():,} fraudes)...")
    modelo = criar_modelo()
    modelo.fit(X_treino, y_treino)
    avaliar(modelo, X_teste, y_teste)

    # O predict.py classifica uma transação por vez; com 1 thread ele inicia mais rápido
    modelo.set_params(floresta__n_jobs=1)
    CAMINHO_MODELO.parent.mkdir(exist_ok=True)
    joblib.dump(modelo, CAMINHO_MODELO, compress=3)
    tamanho_mb = CAMINHO_MODELO.stat().st_size / 1e6
    print(f"\nModelo salvo em {CAMINHO_MODELO} ({tamanho_mb:.1f} MB) em {time.time() - inicio:.0f}s")


if __name__ == "__main__":
    main()
