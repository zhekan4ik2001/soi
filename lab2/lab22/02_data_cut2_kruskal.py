from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

ALPHA = 0.05
FILE = Path(__file__).with_name("data_cut2.csv")

df = pd.read_csv(FILE, sep=None, engine="python")
df["OBS_VALUE"] = pd.to_numeric(df["OBS_VALUE"], errors="coerce")

# --- Визуализация: boxplot по GEO ---
plt.figure(figsize=(10, 6))
# Используем pivot, чтобы каждая группа была отдельным столбцом для boxplot
box_data = [group["OBS_VALUE"].dropna().values for _, group in df.groupby("GEO", sort=True)]
box_labels = [geo for geo, _ in df.groupby("GEO", sort=True)]

plt.boxplot(box_data, labels=box_labels, patch_artist=True)
plt.title("Распределение показателя по GEO-кодам (OBS_VALUE, 2019)")
plt.xlabel("GEO")
plt.ylabel("Значение показателя")
plt.grid(axis="y", linestyle="--", alpha=0.3)
plt.tight_layout()
plt.show()

# --- Статистический анализ ---
groups = []
print("Набор: data_cut2.csv")
print("Показатель: годовое изменение индекса потребительских цен по группам товаров/услуг, 2019 год")
print("Сравниваем распределения значений между четырьмя GEO-кодами.\n")

for geo, part in df.groupby("GEO", sort=True):
    vals = part["OBS_VALUE"].dropna().to_numpy()
    groups.append(vals)
    print(f"GEO={geo}: n={len(vals)}, медиана={pd.Series(vals).median():.3f}")

if len(groups) < 2 or any(len(g) == 0 for g in groups):
    raise ValueError("Для критерия Краскела—Уоллиса нужны минимум две непустые группы.")

statistic, pvalue = stats.kruskal(*groups)

print("\nКритерий Краскела—Уоллиса")
print("H0: распределения показателя во всех GEO-группах одинаковы (в частности, совпадают их положения).")
print("H1: как минимум одна GEO-группа отличается.")
print(f"Уровень значимости alpha = {ALPHA}")
print(f"H = {statistic:.6f}")
print(f"p-value = {pvalue:.6g}")
if pvalue < ALPHA:
    print("Решение: H0 отвергается. Обнаружено статистически значимое различие между группами.")
    print("Критерий не указывает, какие именно пары групп различаются; для этого нужен post-hoc анализ.")
else:
    print("Решение: нет оснований отвергнуть H0; значимое различие не обнаружено.")

print("\nОбоснование выбора: четыре независимые группы и количественный показатель;")
print("не требуется предположение о нормальности распределений.")
