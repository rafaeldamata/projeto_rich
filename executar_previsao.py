# -*- coding: utf-8 -*-
"""
executar_previsao.py
---------------------
Script de linha de comando que une todas as etapas:

    ticks (CSV) -> barras (período escolhido) -> janela deslizante
    (parâmetros de Richards adaptativos) -> previsão até uma data escolhida

Exemplos de uso:

    # Descrição da série em barras de 1 hora, janela de 200 barras
    python executar_previsao.py --periodo 1H --janela 200

    # Previsão até uma data específica, usando barras de 4 horas
    python executar_previsao.py --periodo 4H --janela 150 \
        --prever-ate "2025-02-10 00:00:00"
"""

import argparse

from richards_forex import (
    carregar_ticks,
    reamostrar_em_barras,
    pipeline_janela_deslizante,
    prever_periodo,
    PERIODOS_VALIDOS,
)


def main():
    parser = argparse.ArgumentParser(
        description="Descrição e previsão de série forex via Função de Richards "
                    "em janelas deslizantes adaptativas."
    )
    parser.add_argument(
        "--csv", default="ticks_eurusd.csv",
        help="Caminho do CSV de ticks (timestamp, bid). Padrão: ticks_eurusd.csv",
    )
    parser.add_argument(
        "--periodo", default="1H", choices=list(PERIODOS_VALIDOS.keys()),
        help="Período da barra: 1M, 15M, 30M, 1H, 4H ou 1D. Padrão: 1H",
    )
    parser.add_argument(
        "--janela", type=int, default=100,
        help="Tamanho N da janela deslizante (em número de barras). Padrão: 100",
    )
    parser.add_argument(
        "--passo", type=int, default=1,
        help="Quantas barras a janela avança a cada iteração. Padrão: 1",
    )
    parser.add_argument(
        "--prever-ate", default=None,
        help="Data/hora final da previsão, ex: '2025-02-10 00:00:00'. "
             "Se omitido, só a descrição (trajetória de parâmetros) é exibida.",
    )
    parser.add_argument(
        "--saida-trajetoria", default="trajetoria_parametros.csv",
        help="Onde salvar a trajetória de parâmetros por janela.",
    )
    parser.add_argument(
        "--saida-previsao", default="previsao.csv",
        help="Onde salvar a previsão, se --prever-ate for usado.",
    )

    args = parser.parse_args()

    print(f"Carregando ticks de: {args.csv}")
    ticks = carregar_ticks(args.csv)

    print(f"Reamostrando em barras de período: {args.periodo}")
    barras = reamostrar_em_barras(ticks, args.periodo)
    print(f"Total de barras geradas: {len(barras)}")

    print(f"Executando janela deslizante (N={args.janela}, passo={args.passo})...")
    trajetoria = pipeline_janela_deslizante(
        barras, tamanho_janela=args.janela, passo=args.passo
    )
    trajetoria.to_csv(args.saida_trajetoria, index=False)
    print(f"Trajetória de parâmetros salva em: {args.saida_trajetoria}")
    print(trajetoria.tail())

    if args.prever_ate:
        print(f"\nPrevendo até: {args.prever_ate}")
        previsao = prever_periodo(
            barras, tamanho_janela=args.janela, data_final_previsao=args.prever_ate
        )
        previsao.to_csv(args.saida_previsao)
        print(f"Previsão salva em: {args.saida_previsao}")
        print(f"Parâmetros da última janela: {previsao.attrs['parametros_richards']}")
        print(f"R² do ajuste na última janela: {previsao.attrs['r2_ajuste_janela']:.4f}")
        print(previsao.head())


if __name__ == "__main__":
    main()
