import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import warnings
import dados

# Suprime avisos de falha de convergência durante o fitting da janela móvel
warnings.filterwarnings('ignore')


# ==============================================================================
# 1. DEFINIÇÃO DA FUNÇÃO DE RICHARDS E DERIVADA
# ==============================================================================
def richards_curve(t, A, K, B, t0, nu, Q=1.0):
    """
    Equação da Curva de Richards Aplicada ao Preço:
    P(t) = A + (K - A) / (1 + Q * exp(-B * (t - t0)))^(1 / nu)
    """
    # Evitar divisões por zero ou estouro de expoente (overflow/underflow)
    arg = -B * (t - t0)
    arg = np.clip(arg, -50, 50)
    
    denominator = (1.0 + Q * np.exp(arg)) ** (1.0 / np.maximum(nu, 1e-5))
    return A + (K - A) / denominator

def richards_derivative(t, A, K, B, t0, nu, Q=1.0):
    """
    Primeira derivada dP/dt: Taxa instantânea de variação do preço.
    """
    arg = -B * (t - t0)
    arg = np.clip(arg, -50, 50)
    exp_term = np.exp(arg)
    base = 1.0 + Q * exp_term
    
    dp_dt = (B * (K - A) * Q * exp_term) / (nu * (base ** ((1.0 / nu) + 1.0)))
    return dp_dt


# ==============================================================================
# 2. PoC 1: VALIDAÇÃO DE ALVO DE PREÇO EM IMPULSOS ISOLADOS
# ==============================================================================
def poc1_validacao_alvo(df, inicio_idx, fim_ajuste_idx, janelas_futuras=100, coluna_preco='Close'):
    """
    Ajusta a Função de Richards em um segmento inicial de impulso e compara o K 
    estimado com a máxima/mínima real atingida no futuro.
    """
    print("=" * 60)
    print("INICIANDO PoC 1: Validação de Alvo de Preço em Impulso Isolado")
    print("=" * 60)
    
    # Segmento de ajuste (início do movimento até confirmação inicial)
    slice_ajuste = df.iloc[inicio_idx:fim_ajuste_idx].copy()
    y_train = slice_ajuste[coluna_preco].values
    x_train = np.arange(len(y_train))
    
    # Segmento futuro real para comparação
    fim_futuro_idx = min(fim_ajuste_idx + janelas_futuras, len(df))
    slice_futuro = df.iloc[inicio_idx:fim_futuro_idx].copy()
    y_futuro_real = slice_futuro[coluna_preco].values
    x_futuro = np.arange(len(y_futuro_real))
    
    # Valoração inicial dos parâmetros (Chute Inicial / Bounds)
    A_init = y_train[0]
    K_init = np.max(y_train) if y_train[-1] >= y_train[0] else np.min(y_train)
    t0_init = len(y_train) / 2.0
    p0 = [A_init, K_init, 0.1, t0_init, 1.0]
    
    # Ajuste por mínimos quadrados não lineares (Levenberg-Marquardt / Trust Region)
    try:
        popt, pcov = curve_fit(
            richards_curve, x_train, y_train, p0=p0, 
            maxfev=10000, bounds=([-np.inf, -np.inf, 0.0001, -np.inf, 0.0001], [np.inf, np.inf, 10.0, np.inf, 50.0])
        )
        A_est, K_est, B_est, t0_est, nu_est = popt
        
        max_real = np.max(y_futuro_real)
        min_real = np.min(y_futuro_real)
        alvo_real = max_real if y_train[-1] >= y_train[0] else min_real
        erro_absoluto = abs(K_est - alvo_real)
        erro_percentual = (erro_absoluto / alvo_real) * 100

        print(f"Parâmetros Ajustados:")
        print(f"  A (Nível Inicial) = {A_est:.5f}")
        print(f"  K (Assíntota / Alvo Estimado) = {K_est:.5f}")
        print(f"  B (Taxa de Velocidade)        = {B_est:.5f}")
        print(f"  t0 (Ponto de Inflexão)       = {t0_est:.2f}")
        print(f"  ν (Parâmetro de Assimetria)   = {nu_est:.5f}")
        print("-" * 50)
        print(f"Alvo Real Atingido no Futuro   = {alvo_real:.5f}")
        print(f"Erro de Estimativa do Alvo     = {erro_absoluto:.5f} ({erro_percentual:.3f}%)")
        
        # Plotagem dos resultados da PoC 1
        plt.figure(figsize=(12, 6))
        plt.plot(x_futuro, y_futuro_real, label='Preço Real (Futuro)', color='black', alpha=0.7)
        plt.plot(x_train, y_train, label='Dados do Ajuste (Treino)', color='blue', linewidth=2.5)
        plt.plot(x_futuro, richards_curve(x_futuro, *popt), label='Curva de Richards Projetada', color='red', linestyle='--')
        plt.axhline(y=K_est, color='green', linestyle=':', label=f'Alvo Estimado K ({K_est:.5f})')
        plt.axhline(y=alvo_real, color='orange', linestyle=':', label=f'Alvo Real Atingido ({alvo_real:.5f})')
        
        plt.title('PoC 1: Ajuste da Curva de Richards e Projeção de Alvo K')
        plt.xlabel('Ticks / Unidades de Tempo')
        plt.ylabel('Preço')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.show()

    except Exception as e:
        print(f"Falha ao ajustar a curva na PoC 1: {e}")


# ==============================================================================
# 3. PoC 2: ENGENHARIA DE FEATURES EM JANELA MÓVEL (ROLLING WINDOW)
# ==============================================================================
            
    


# ==============================================================================
# 4. EXECUÇÃO DO EXPERIMENTO (SUBSTITUA PELOS SEUS DADOS)
# ==============================================================================
if __name__ == '__main__':
    # EXEMPLE: Carregando seus dados salvos do EUR/USD (15m ou Ticks)
    # df_forex = pd.read_csv("eurusd_2024_15m.csv")
    
    # --- SIMULAÇÃO DE DADOS DE TESTE (Remova esta bloco ao carregar seu CSV) ---
    df_forex = dados.baixar_dados()
    # -------------------------------------------------------------------------

    # 1. Executar PoC 1
    # Define um impulso: ex. ajusta dos pontos 200 até 450 e projeta os próximos 150 pontos
    poc1_validacao_alvo(df_forex, inicio_idx=200, fim_ajuste_idx=450, janelas_futuras=150, coluna_preco='Close')
    
    # 2. Executar PoC 2
    # Roda a janela móvel de 200 em 200 pontos com passo de 5
    df_features = poc2_rolling_features(df_forex, tamanho_janela=200, passo=5, horizonte_futuro=30, coluna_preco='Close')