from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Настройка стиля графиков
plt.style.use('seaborn-v0_8-whitegrid')

ALPHA = 0.05
FILE = Path(__file__).with_name("data_cut3.csv")

df = pd.read_csv(FILE, sep=None, engine="python")
df["OBS_VALUE"] = pd.to_numeric(df["OBS_VALUE"], errors="coerce")

wide = df.pivot_table(index="GEO", columns="TPH", values="OBS_VALUE", aggfunc="mean")
required = ["11", "MF22"]
missing = [col for col in required if col not in wide.columns]
if missing:
    raise ValueError(f"В данных отсутствуют категории TPH: {missing}")

paired = wide[required].dropna()
x = paired["11"].to_numpy()
y = paired["MF22"].to_numpy()

if len(x) == 0:
    raise ValueError("Не найдено ни одной полной пары GEO.")

# Wilcoxon не принимает ситуацию, когда все парные разности равны нулю.
if (x == y).all():
    raise ValueError("Все парные различия равны нулю; критерий не может быть рассчитан обычным способом.")


# --- Визуализация данных ---
fig, axs = plt.subplots(1, 2, figsize=(14, 5))

# 1. Точечная диаграмма парных значений
axs[0].scatter(x, y, alpha=0.7, edgecolors='k', linewidth=0.5)
axs[0].plot([x.min(), x.max()], [x.min(), x.max()], 'r--', label='Линия равенства')
axs[0].set_xlabel('TPH = 11 (среднее число жилищ)')
axs[0].set_ylabel('TPH = MF22 (среднее число жилищ)')
axs[0].set_title('Парные значения по территориям (GEO)')
axs[0].legend()

# 2. Ящик с усами для сравнения распределений
axs[1].boxplot([x, y], labels=['TPH = 11', 'TPH = MF22'], patch_artist=True)
axs[1].set_ylabel('Число основных жилищ')
axs[1].set_title('Распределение значений по категориям TPH')

plt.tight_layout()
plt.show()
# -------------------------

statistic, pvalue = stats.wilcoxon(x, y, alternative="two-sided", zero_method="wilcox")

print("Набор: data_cut3.csv")
print("Показатель: число основных жилищ по двум категориям TPH в 2023 году")
print(f"Число парных территорий: n = {len(paired)}")
print(f"Медиана TPH=11: {pd.Series(x).median():.3f}")
print(f"Медиана TPH=MF22: {pd.Series(y).median():.3f}")
print("\nКритерий знаковых рангов Уилкоксона")
print("H0: медиана парных различий между TPH=11 и TPH=MF22 равна нулю.")
print("H1: медиана парных различий отличается от нуля.")
print(f"Уровень значимости alpha = {ALPHA}")
print(f"W = {statistic:.6f}")
print(f"p-value = {pvalue:.6g}")
if pvalue < ALPHA:
    print("Решение: H0 отвергается. Парные значения статистически значимо различаются.")
else:
    print("Решение: нет оснований отвергнуть H0; значимое парное различие не обнаружено.")

print("\nОбоснование выбора: для каждой GEO-территории есть оба значения, поэтому наблюдения связаны.")
print("Критерий Уилкоксона — непараметрическая альтернатива парному t-критерию.")
