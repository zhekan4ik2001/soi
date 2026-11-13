from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Настройка стиля графиков
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("colorblind")

ALPHA = 0.05
FILE = Path(__file__).with_name("data_cut4.csv")

df = pd.read_csv(FILE, sep=None, engine="python")
df["OBS_VALUE"] = pd.to_numeric(df["OBS_VALUE"], errors="coerce")

# D и C в единицах XDC; строки PT исключаем, поскольку это другая единица измерения.
sub = df[df["UNIT_MEASURE"].eq("XDC") & df["ACCOUNTING_ENTRY"].isin(["D", "C"])].copy()
d_values = sub.loc[sub["ACCOUNTING_ENTRY"].eq("D"), "OBS_VALUE"].dropna().to_numpy()
c_values = sub.loc[sub["ACCOUNTING_ENTRY"].eq("C"), "OBS_VALUE"].dropna().to_numpy()

if len(d_values) == 0 or len(c_values) == 0:
    raise ValueError("После фильтрации нет данных для одной из групп.")


# --- Визуализация данных ---
fig, axs = plt.subplots(1, 2, figsize=(14, 5))

# 1. Гистограммы + KDE (оценка плотности распределения)
sns.histplot(d_values, kde=True, ax=axs[0], label="ACCOUNTING_ENTRY = D", color="#1f77b4")
sns.histplot(c_values, kde=True, ax=axs[0], label="ACCOUNTING_ENTRY = C", color="#d62728")
axs[0].set_xlabel("OBS_VALUE (единицы XDC)")
axs[0].set_ylabel("Частота / плотность")
axs[0].set_title("Распределения значений групп D и C (гистограмма + KDE)")
axs[0].legend()

# 2. Ящики с усами (медианы, разброс, выбросы)
sns.boxplot(data=[d_values, c_values], ax=axs[1], palette=["#1f77b4", "#d62728"])
axs[1].set_xticklabels(["ACCOUNTING_ENTRY = D", "ACCOUNTING_ENTRY = C"])
axs[1].set_ylabel("OBS_VALUE (единицы XDC)")
axs[1].set_title("Сравнение групп D и C: медианы и разброс (boxplot)")

plt.tight_layout()
plt.show()
# -------------------------

statistic, pvalue = stats.mannwhitneyu(
    d_values, c_values, alternative="two-sided", method="auto"
)

print("Набор: data_cut4.csv")
print("Показатель: значения национальных счетов Франции в единицах XDC, 2019 год")
print("Сравниваются группы ACCOUNTING_ENTRY=D и ACCOUNTING_ENTRY=C.")
print("Внимание: это сравнение распределений строк двух типов, а не доказательство равенства/неравенства")
print("экономических итогов без дополнительной проверки сопоставимости строк.\n")
print(f"D: n={len(d_values)}, медиана={pd.Series(d_values).median():.3f}")
print(f"C: n={len(c_values)}, медиана={pd.Series(c_values).median():.3f}")
print("\nU-критерий Манна—Уитни")
print("H0: распределения значений в группах D и C одинаковы.")
print("H1: распределения значений в группах D и C различаются.")
print(f"Уровень значимости alpha = {ALPHA}")
print(f"U = {statistic:.6f}")
print(f"p-value = {pvalue:.6g}")
if pvalue < ALPHA:
    print("Решение: H0 отвергается. Распределения значений статистически значимо различаются.")
else:
    print("Решение: нет оснований отвергнуть H0; значимое различие не обнаружено.")

print("\nОбоснование выбора: сравниваются две независимые группы количественных значений,")
print("при этом не предполагается нормальность распределения.")
