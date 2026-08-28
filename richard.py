import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# 1. Carregar os dados do arquivo CSV
df = pd.read_csv('ticks_tempo_bid.csv')
t_data = df['Tempo (t)'].values
bid_data = df['Bid'].values

# 2. Definição da Função de Richards com o parâmetro Q explícito
def richards(t, A, K, B, t0, nu, Q):
    """
    Modelo de Richards:
    P(t) = A + (K - A) / (1 + Q * exp(-B * (t - t0)))^(1 / nu)
    """
    return A + (K - A) / ((1 + Q * np.exp(-B * (t - t0))) ** (1 / nu))

# 3. Estimativa inicial dos parâmetros [A, K, B, t0, nu, Q]
p0 = [min(bid_data), max(bid_data), 0.12, 45.0, 0.85, 1.0]

# Limites de busca (bounds) para garantir estabilidade numérica no Forex
bounds = (
    [1.0700, 1.0800, 0.001, 1.0, 0.01, 0.01],    # Limites inferiores
    [1.0900, 1.1000, 2.000, 100.0, 10.0, 100.0]   # Limites superiores
)

# 4. Execução da Regressão Não Linear
popt, pcov = curve_fit(
    richards, 
    t_data, 
    bid_data, 
    p0=p0, 
    bounds=bounds, 
    method='trf', 
    maxfev=20000
)

A_opt, K_opt, B_opt, t0_opt, nu_opt, Q_opt = popt

# 5. Cálculo dos valores previstos e métricas de acerto
bid_pred = richards(t_data, *popt)

r2 = r2_score(bid_data, bid_pred)
rmse = np.sqrt(mean_squared_error(bid_data, bid_pred))
mae = mean_absolute_error(bid_data, bid_pred)

# 6. Exibição dos Parâmetros Otimizados
print("=" * 50)
print(" RESULTADOS DA REGRESSÃO NÃO LINEAR (RICHARDS) ")
print("=" * 50)
print(f" A (Suporte Mínimo)  : {A_opt:.6f}")
print(f" K (Resistência Máx) : {K_opt:.6f}")
print(f" B (Taxa de Variação): {B_opt:.6f}")
print(f" t0 (Ponto Inflexão) : {t0_opt:.6f}")
print(f" nu (Assimetria)     : {nu_opt:.6f}")
print(f" Q (Deslocamento)    : {Q_opt:.6f}")
print("-" * 50)
print(f" Acurácia R²         : {r2 * 100:.6f}%")
print(f" Erro RMSE           : {rmse:.8f}")
print(f" Erro MAE            : {mae:.8f}")
print("=" * 50)

# 7. Visualização Gráfica da Regressão
plt.figure(figsize=(10, 6), dpi=150)
plt.plot(t_data, bid_data, 'o', color='#1f77b4', label='Ticks Bid Observados', markersize=4, alpha=0.7)
plt.plot(t_data, bid_pred, '-', color='#d62728', label=f'Curva de Richards ($R^2 = {r2*100:.4f}\\%$)', linewidth=2)

plt.title('Regressão Não Linear - Ajuste dos Ticks Forex com Função de Richards', fontsize=12, fontweight='bold')
plt.xlabel('Tempo ($t$ / Ticks)')
plt.ylabel('Preço Bid')
plt.gca().yaxis.set_major_formatter('{x:.5f}')
plt.legend(loc='upper left')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()