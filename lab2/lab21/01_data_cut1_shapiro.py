from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np

ALPHA = 0.05
FILE = Path(__file__).with_name("data_cut1.csv")

df = pd.read_csv(FILE, sep=None, engine="python")
df["OBS_VALUE"] = pd.to_numeric(df["OBS_VALUE"], errors="coerce")

# В выборке оставляем территориальные единицы типа DEP, не смешивая их
# с региональными итогами и прочими агрегатами.
sample = df.loc[df["GEO_OBJECT"].eq("DEP"), "OBS_VALUE"].dropna().to_numpy()

if len(sample) < 3:
    raise ValueError("Для критерия Шапиро—Уилка нужно не менее 3 наблюдений.")

# --- Визуализация ---
plt.style.use("seaborn-v0_8-whitegrid")
fig, axs = plt.subplots(1, 2, figsize=(14, 5))

# Гистограмма + плотность нормального распределения
sns.histplot(sample, kde=False, stat="density", ax=axs[0], color="#4c72b0")
# Подгоняем нормальное распределение по выборочным среднему и СКО
mu, sigma = np.mean(sample), np.std(sample, ddof=0)
x = np.linspace(np.min(sample), np.max(sample), 200)
axs[0].plot(x, stats.norm.pdf(x, mu, sigma), color="#d62728", linewidth=2, label="Нормальное распределение")
axs[0].set_title("Гистограмма и плотность нормального распределения")
axs[0].set_xlabel("Число смертей по департаментам")
axs[0].set_ylabel("Плотность")
axs[0].legend()

# Q-Q график
stats.probplot(sample, dist="norm", plot=axs[1])
axs[1].set_title("Q-Q график (нормальность)")
axs[1].set_xlabel("Теоретические квантили")
axs[1].set_ylabel("Выборочные квантили")

plt.tight_layout()
plt.show()

# --- Статистика ---
statistic, pvalue = stats.shapiro(sample)

print("Набор: data_cut1.csv")
print("Показатель: число смертей по департаментам Франции, 2019 год")
print(f"Число наблюдений: n = {len(sample)}")
print(f"Среднее: {sample.mean():.3f}; медиана: {pd.Series(sample).median():.3f}")
print("\nКритерий Шапиро—Уилка")
print("H0: значения выборки происходят из нормально распределённого генерального распределения.")
print("H1: распределение отличается от нормального.")
print(f"Уровень значимости alpha = {ALPHA}")
print(f"W = {statistic:.6f}")
print(f"p-value = {pvalue:.6g}")
if pvalue < ALPHA:
    print("Решение: H0 отвергается. Есть статистические основания считать распределение ненормальным.")
else:
    print("Решение: нет оснований отвергнуть H0; нормальность не опровергнута.")

print("\nПримечание: в презентации критерий описан прежде всего для небольших выборок.")
print("Здесь n может быть больше 50; SciPy допускает применение Shapiro—Wilk для таких объёмов,")
print("но результат стоит интерпретировать вместе с гистограммой и Q-Q графиком.")
