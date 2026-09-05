# -*- coding: utf-8 -*-
"""
richards_forex.py
------------------
Biblioteca para descrição e previsão de uma série temporal forex (EUR/USD)
usando a Função de Crescimento de Richards ajustada em janelas deslizantes
adaptativas.

Fluxo geral (pipeline):
    1. carregar_ticks()            -> lê o CSV de ticks brutos
    2. reamostrar_em_barras()      -> agrega os ticks no período de barra
                                       escolhido (1M, 15M, 30M, 1H, 4H, 1D)
    3. ajustar_janela_richards()   -> ajusta os 5 parâmetros de Richards
                                       (A, K, B, M, nu) em UMA janela
    4. pipeline_janela_deslizante()-> percorre toda a série em janelas de
                                       tamanho N, reajustando os parâmetros
                                       a cada passo (parâmetros adaptativos)
    5. prever_periodo()            -> usa a última janela disponível (ou uma
                                       janela escolhida) para extrapolar a
                                       curva de Richards até a data alvo
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


# ---------------------------------------------------------------------------
# 1) Mapeamento dos períodos de barra aceitos pelo usuário
# ---------------------------------------------------------------------------
PERIODOS_VALIDOS = {
    "1M": "1min",   # 1 minuto
    "15M": "15min", # 15 minutos
    "30M": "30min", # 30 minutos
    "1H": "1h",     # 1 hora
    "4H": "4h",     # 4 horas
    "1D": "1D",     # 1 dia
}


def carregar_ticks(caminho_csv: str) -> pd.DataFrame:
    """Lê o CSV de ticks (timestamp, bid) e devolve um DataFrame indexado
    pelo timestamp, pronto para reamostragem."""
    df = pd.read_csv(caminho_csv, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").set_index("timestamp")
    return df


def reamostrar_em_barras(df_ticks: pd.DataFrame, periodo: str) -> pd.DataFrame:
    """
    Agrega os ticks (coluna 'bid') no período de barra escolhido.

    periodo: uma das chaves de PERIODOS_VALIDOS -> '1M','15M','30M','1H','4H','1D'

    Retorna um DataFrame com colunas open/high/low/close, uma linha por barra.
    O preço usado para o ajuste de Richards é a coluna 'close'.
    """
    if periodo not in PERIODOS_VALIDOS:
        raise ValueError(
            f"Período '{periodo}' inválido. Use um de: {list(PERIODOS_VALIDOS)}"
        )
    freq = PERIODOS_VALIDOS[periodo]
    ohlc = df_ticks["bid"].resample(freq).ohlc()
    ohlc = ohlc.dropna()  # remove barras sem nenhum tick (ex.: fim de semana)
    return ohlc


# ---------------------------------------------------------------------------
# 2) Função de Richards
# ---------------------------------------------------------------------------
def richards(t, A, K, B, M, nu):
    """
    Função de crescimento de Richards (generalização da logística):

        y(t) = A + (K - A) / (1 + nu * exp(-B * (t - M)))^(1/nu)

    Parâmetros:
        A  -> assíntota inferior (nível de partida)
        K  -> assíntota superior (nível de saturação/chegada)
        B  -> taxa de crescimento (velocidade da transição)
        M  -> tempo do ponto de inflexão (onde a curva muda de concavidade)
        nu -> parâmetro de forma/assimetria (nu=1 recai na logística clássica)

    t pode ser escalar ou array (índice de barra dentro da janela: 0,1,2,...).
    """
    nu_seguro = np.clip(nu, 1e-6, None)
    return A + (K - A) / np.power(1.0 + nu_seguro * np.exp(-B * (t - M)), 1.0 / nu_seguro)


@dataclass
class ResultadoAjuste:
    params: np.ndarray          # [A, K, B, M, nu]
    sucesso: bool
    r2: float
    fim_janela: pd.Timestamp    # timestamp da última barra da janela


def ajustar_janela_richards(
    y: np.ndarray,
    p0: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, bool, float]:
    """
    Ajusta os parâmetros da Função de Richards a UMA janela de N observações.

    y: array 1D com os preços de fechamento da janela (tamanho N)
    p0: chute inicial [A, K, B, M, nu]. Se None, é estimado a partir dos
        próprios dados da janela. Passar o p0 = parâmetros da janela
        anterior é o que torna o ajuste "adaptativo" e mais estável
        (warm start) de uma janela para a seguinte.

    Retorna: (parametros_ajustados, sucesso, r2)
    """
    n = len(y)
    t = np.arange(n, dtype=float)

    if p0 is None:
        A0 = y[0]
        K0 = y[-1]
        M0 = n / 2.0
        B0 = 0.1 if K0 >= A0 else -0.1
        nu0 = 1.0
        p0 = [A0, K0, B0, M0, nu0]

    # Limites (bounds) para manter o ajuste em uma região com sentido físico:
    # A, K -> próximos da faixa observada de preços (com folga)
    faixa = max(y.max() - y.min(), 1e-6)
    lim_inf = [y.min() - faixa, y.min() - faixa, -5.0, -n, 0.01]
    lim_sup = [y.max() + faixa, y.max() + faixa, 5.0, 2 * n, 10.0]

    # Quando p0 vem da janela anterior (warm start), ele foi estimado com
    # bounds de OUTRA janela (outra faixa de preço/tempo) e pode cair fora
    # dos bounds desta nova janela -- o que faria o curve_fit falhar de
    # cara com "x0 infeasible". Por isso, sempre "recolocamos" o p0 dentro
    # dos limites atuais antes de chamar o otimizador.
    p0 = np.clip(np.asarray(p0, dtype=float), lim_inf, lim_sup)

    try:
        params_ajustados, _ = curve_fit(
            richards, t, y, p0=p0, bounds=(lim_inf, lim_sup), maxfev=20000
        )
        y_pred = richards(t, *params_ajustados)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        return params_ajustados, True, r2
    except Exception:
        # Se o ajuste não convergir nesta janela, devolve o p0 como fallback
        # e sinaliza sucesso=False (o chamador decide o que fazer).
        return np.array(p0), False, np.nan


def pipeline_janela_deslizante(
    barras: pd.DataFrame,
    tamanho_janela: int,
    passo: int = 1,
    coluna_preco: str = "close",
) -> pd.DataFrame:
    """
    Percorre toda a série de barras em janelas deslizantes de tamanho
    `tamanho_janela`, avançando `passo` barras a cada iteração, e reajusta
    os parâmetros de Richards EM CADA janela (parâmetros adaptativos).

    Usa o ajuste da janela anterior como chute inicial (warm start) da
    janela seguinte, para dar continuidade suave à trajetória dos
    parâmetros.

    Retorna um DataFrame com uma linha por janela, contendo:
        fim_janela, A, K, B, M, nu, r2, sucesso
    Essa tabela é a "trajetória no espaço paramétrico" mencionada no
    projeto: cada linha é um vetor (A,K,B,M,nu) que descreve o regime
    vigente naquele trecho da série.
    """
    precos = barras[coluna_preco].to_numpy()
    tempos = barras.index

    registros = []
    p0_anterior = None

    for inicio in range(0, len(precos) - tamanho_janela + 1, passo):
        fim = inicio + tamanho_janela
        janela_y = precos[inicio:fim]

        params, sucesso, r2 = ajustar_janela_richards(janela_y, p0=p0_anterior)

        # só usamos o ajuste atual como warm start do próximo se ele convergiu
        if sucesso:
            p0_anterior = params

        A, K, B, M, nu = params
        registros.append(
            {
                "fim_janela": tempos[fim - 1],
                "A": A, "K": K, "B": B, "M": M, "nu": nu,
                "r2": r2,
                "sucesso": sucesso,
            }
        )

    return pd.DataFrame(registros)


def prever_periodo(
    barras: pd.DataFrame,
    tamanho_janela: int,
    data_final_previsao: str,
    coluna_preco: str = "close",
) -> pd.DataFrame:
    """
    Faz a previsão real (fora da amostra) da série a partir da ÚLTIMA janela
    disponível de tamanho `tamanho_janela`:

        1. Pega as últimas `tamanho_janela` barras conhecidas.
        2. Ajusta a Função de Richards nessa janela (t local = 0..N-1).
        3. Extrapola a curva ajustada para os instantes futuros, do fim da
           janela até `data_final_previsao`, usando o mesmo período de
           barra da série de entrada.

    data_final_previsao: string de data/hora (ex.: '2025-02-10 00:00:00')
        até onde se quer projetar a previsão.

    Retorna um DataFrame com colunas: timestamp, previsao, alem_da_janela=True
    """
    freq = barras.index.freq or pd.infer_freq(barras.index)
    if freq is None:
        raise ValueError(
            "Não foi possível inferir a frequência das barras; "
            "gere as barras com reamostrar_em_barras() para preservar a frequência."
        )

    if len(barras) < tamanho_janela:
        raise ValueError("A série tem menos barras que o tamanho da janela.")

    ultima_janela = barras.iloc[-tamanho_janela:]
    y = ultima_janela[coluna_preco].to_numpy()

    params, sucesso, r2 = ajustar_janela_richards(y)
    if not sucesso:
        raise RuntimeError("O ajuste de Richards não convergiu na última janela.")

    ultimo_timestamp = ultima_janela.index[-1]
    datas_futuras = pd.date_range(
        start=ultimo_timestamp, end=data_final_previsao, freq=freq
    )[1:]  # exclui o próprio último timestamp conhecido

    if len(datas_futuras) == 0:
        raise ValueError(
            "A data final de previsão deve ser posterior à última barra conhecida."
        )

    # t=N-1 corresponde ao último ponto conhecido da janela; os pontos
    # futuros continuam essa contagem (N, N+1, N+2, ...)
    n = tamanho_janela
    t_futuro = np.arange(n, n + len(datas_futuras), dtype=float)
    y_previsto = richards(t_futuro, *params)

    resultado = pd.DataFrame(
        {"timestamp": datas_futuras, "previsao": y_previsto}
    ).set_index("timestamp")

    resultado.attrs["parametros_richards"] = dict(zip("A K B M nu".split(), params))
    resultado.attrs["r2_ajuste_janela"] = r2
    return resultado
