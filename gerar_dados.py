# -*- coding: utf-8 -*-
"""
gerar_dados.py
--------------
Gera uma série fictícia (sintética) de ticks do par EUR/USD e salva em CSV.

A série NÃO é um passeio aleatório puro: ela é construída como uma soma de
vários "regimes" com formato sigmoide (S) + ruído, exatamente para que,
quando reamostrada em barras (1M, 15M, 30M, 1H, 4H, 1D) e analisada em
janelas deslizantes, a Função de Richards tenha padrões de tendência/
saturação que façam sentido ajustar.

Saída: ticks_eurusd.csv com colunas:
    timestamp (datetime ISO)
    bid (float)
"""

import numpy as np
import pandas as pd

def richards(t, A, K, B, M, nu):
    """Função de crescimento de Richards (usada aqui só para 'desenhar' regimes)."""
    nu = max(nu, 1e-6)
    return A + (K - A) / np.power(1.0 + nu * np.exp(-B * (t - M)), 1.0 / nu)


def gerar_ticks_eurusd(
    inicio="2025-01-01 00:00:00",
    n_ticks=200_000,
    freq_segundos=15,
    preco_base=1.0850,
    seed=42,
    caminho_saida="ticks_eurusd.csv",
):
    """
    Gera n_ticks ticks espaçados por freq_segundos segundos, combinando:
      - vários trechos de "regime" com formato de Richards (tendências que
        aceleram, desaceleram e saturam);
      - ruído gaussiano de microestrutura (o "chiado" natural do tick a tick);
      - pequena reversão à média para não deixar a série "fugir" para longe
        de um patamar plausível de EUR/USD.
    """
    rng = np.random.default_rng(seed)

    # eixo de tempo em "índice de tick" (0, 1, 2, ...) -- será convertido em
    # timestamps reais no final
    t = np.arange(n_ticks, dtype=float)

    # 1) Construção dos regimes de Richards (tendências macro) ---------------
    # Dividimos a série em blocos e, em cada bloco, sorteamos parâmetros de
    # Richards que definem uma transição suave de um nível de preço a outro.
    n_blocos = 12
    limites = np.linspace(0, n_ticks, n_blocos + 1).astype(int)

    nivel_atual = preco_base
    serie_regime = np.zeros(n_ticks)

    for i in range(n_blocos):
        ini, fim = limites[i], limites[i + 1]
        n_local = fim - ini
        t_local = np.arange(n_local, dtype=float)

        # variação total do bloco (em pips, convertida para preço): entre
        # -80 e +80 pips, ou seja, entre -0.0080 e +0.0080
        delta = rng.uniform(-0.0080, 0.0080)
        novo_nivel = nivel_atual + delta

        # parâmetros de Richards para essa transição:
        A_ = nivel_atual                       # nível inicial do bloco
        K_ = novo_nivel                        # nível final (assíntota) do bloco
        M_ = n_local * rng.uniform(0.3, 0.7)    # ponto de inflexão dentro do bloco
        B_ = rng.uniform(0.01, 0.05) / max(n_local, 1) * n_local * 0.02  # taxa de crescimento
        B_ = rng.uniform(3.0, 8.0) / n_local    # taxa de crescimento (escalada ao tamanho do bloco)
        nu_ = rng.uniform(0.3, 3.0)             # assimetria da curva (parâmetro de forma)

        serie_regime[ini:fim] = richards(t_local, A_, K_, B_, M_, nu_)
        nivel_atual = novo_nivel

    # 2) Ruído de microestrutura (tick a tick) --------------------------------
    ruido = rng.normal(loc=0.0, scale=0.00004, size=n_ticks)  # ~0.4 pip de desvio padrão

    # 3) Pequena correção de reversão à média para manter realismo -----------
    preco = serie_regime + np.cumsum(ruido) * 0.15 + ruido

    # garante que não fique com valores absurdos (proteção simples)
    preco = np.clip(preco, preco_base - 0.05, preco_base + 0.05)

    # 4) Monta timestamps reais -----------------------------------------------
    timestamps = pd.date_range(
        start=inicio, periods=n_ticks, freq=f"{freq_segundos}s"
    )

    df = pd.DataFrame({"timestamp": timestamps, "bid": np.round(preco, 5)})
    df.to_csv(caminho_saida, index=False)
    print(f"Arquivo gerado: {caminho_saida} ({len(df)} ticks, "
          f"de {df['timestamp'].iloc[0]} até {df['timestamp'].iloc[-1]})")
    return df


if __name__ == "__main__":
    gerar_ticks_eurusd()
