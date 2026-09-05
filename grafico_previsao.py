# -*- coding: utf-8 -*-
"""
grafico_previsao.py
--------------------
Gera um gráfico (PNG) mostrando:
    1. O histórico de preços (barras) reamostrado no período escolhido;
    2. A curva de Richards ajustada sobre a última janela de tamanho N
       (a "descrição" do regime mais recente);
    3. A projeção dessa curva para além da janela, até a data escolhida
       (a "previsão" propriamente dita).

Uso:
    python grafico_previsao.py --periodo 1H --janela 100 \
        --prever-ate "2025-02-10 00:00:00" --historico 300
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np

from richards_forex import (
    carregar_ticks,
    reamostrar_em_barras,
    ajustar_janela_richards,
    richards,
    PERIODOS_VALIDOS,
)


def plotar_projecao(
    barras,
    tamanho_janela: int,
    data_final_previsao: str,
    n_barras_historico: int = 300,
    caminho_saida: str = "grafico_previsao_richards.png",
):
    freq = barras.index.freq or barras.index.inferred_freq
    if freq is None:
        raise ValueError("Não foi possível inferir a frequência das barras.")

    if len(barras) < tamanho_janela:
        raise ValueError("A série tem menos barras que o tamanho da janela.")

    # --- ajusta a última janela conhecida ------------------------------
    ultima_janela = barras.iloc[-tamanho_janela:]
    y_janela = ultima_janela["close"].to_numpy()
    params, sucesso, r2 = ajustar_janela_richards(y_janela)
    if not sucesso:
        raise RuntimeError("O ajuste de Richards não convergiu na última janela.")

    # curva ajustada dentro da janela (t = 0..N-1)
    t_janela = np.arange(tamanho_janela, dtype=float)
    y_ajustado = richards(t_janela, *params)

    # --- projeção futura -------------------------------------------------
    import pandas as pd
    ultimo_timestamp = ultima_janela.index[-1]
    datas_futuras = pd.date_range(
        start=ultimo_timestamp, end=data_final_previsao, freq=freq
    )[1:]
    if len(datas_futuras) == 0:
        raise ValueError("A data final de previsão deve ser posterior à última barra.")

    t_futuro = np.arange(
        tamanho_janela, tamanho_janela + len(datas_futuras), dtype=float
    )
    y_futuro = richards(t_futuro, *params)

    # --- trecho de histórico adicional (para dar contexto no gráfico) ---
    n_contexto = max(n_barras_historico - tamanho_janela, 0)
    contexto = barras.iloc[-(tamanho_janela + n_contexto):-tamanho_janela]

    # --- monta o gráfico ---------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 6))

    if len(contexto) > 0:
        ax.plot(
            contexto.index, contexto["close"],
            color="#888888", linewidth=1, label="Histórico (fora da janela)",
        )

    ax.plot(
        ultima_janela.index, y_janela,
        color="#1f77b4", linewidth=1.2, label="Histórico (janela ajustada)",
    )

    ax.plot(
        ultima_janela.index, y_ajustado,
        color="#ff7f0e", linewidth=2.2, linestyle="--",
        label=f"Curva de Richards ajustada (R²={r2:.4f})",
    )

    ax.plot(
        datas_futuras, y_futuro,
        color="#d62728", linewidth=2.2,
        label="Projeção de Richards (fora da amostra)",
    )

    ax.axvline(
        ultimo_timestamp, color="black", linestyle=":", linewidth=1,
        label="Última barra conhecida",
    )

    ax.set_title("Descrição e projeção da série EUR/USD via Função de Richards")
    ax.set_xlabel("Data/Hora")
    ax.set_ylabel("Preço (bid)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(caminho_saida, dpi=150)
    plt.close(fig)

    print(f"Gráfico salvo em: {caminho_saida}")
    print(f"Parâmetros da última janela: "
          f"A={params[0]:.5f}, K={params[1]:.5f}, B={params[2]:.5f}, "
          f"M={params[3]:.2f}, nu={params[4]:.3f}")
    print(f"R² do ajuste na última janela: {r2:.4f}")
    return caminho_saida


def main():
    parser = argparse.ArgumentParser(
        description="Gera gráfico da projeção de Richards a partir do CSV de ticks."
    )
    parser.add_argument("--csv", default="ticks_eurusd.csv")
    parser.add_argument("--periodo", default="1H", choices=list(PERIODOS_VALIDOS.keys()))
    parser.add_argument("--janela", type=int, default=100)
    parser.add_argument("--prever-ate", required=True,
                         help="Data/hora final da previsão, ex: '2025-02-10 00:00:00'")
    parser.add_argument("--historico", type=int, default=300,
                         help="Nº de barras de histórico exibidas antes da janela (contexto visual)")
    parser.add_argument("--saida", default="grafico_previsao_richards.png")
    args = parser.parse_args()

    ticks = carregar_ticks(args.csv)
    barras = reamostrar_em_barras(ticks, args.periodo)
    plotar_projecao(
        barras,
        tamanho_janela=args.janela,
        data_final_previsao=args.prever_ate,
        n_barras_historico=args.historico,
        caminho_saida=args.saida,
    )


if __name__ == "__main__":
    main()
