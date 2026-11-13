import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# --- Чтение и очистка данных ---
df = pd.read_csv("data.csv", sep=";")

df["value"] = pd.to_numeric(
    df["value"].astype(str).str.replace(",", ".").str.replace(" ", ""),
    errors="coerce"
)
df = df.dropna(subset=["value"])

# --- Числовые характеристики ---
print("Статистика по первым 15 записям:")
print(df.head(15)["value"].describe(), "\n")

print("Статистика по всему столбцу value:")
print(df["value"].describe(), "\n")

v = df["value"]
print(f"Медиана:        {v.median():.2f}")
print(f"Дисперсия:      {v.var():.2f}")
print(f"Стандартное отклонение: {v.std():.2f}")
print(f"Коэффициент вариации:   {v.std() / v.mean():.2%}")
print(f"Пропущенных значений:   {df['value'].isna().sum()}\n")

# --- Столбчатая диаграмма: первые 15 записей ---
data = df.head(15)

plt.figure(figsize=(12, 6))
plt.bar(data["reporter_area_label"], data["value"],
        color="steelblue", edgecolor="black")
plt.title("domestic - accommodation - commercial - guests")
plt.xlabel("Страна")
plt.ylabel("Значение")
plt.xticks(rotation=45, ha="right")
plt.grid(axis="y", alpha=0.5)
plt.tight_layout()
plt.savefig("bar_chart.png", dpi=150)
plt.show()

# --- Оценка плотности распределения (гистограмма + ядерная оценка) ---
plt.figure(figsize=(12, 6))
counts, bin_edges, patches = plt.hist(v, bins=15, density=True,
                                      color="lightsteelblue",
                                      edgecolor="black",
                                      label="Гистограмма плотности")

kde = stats.gaussian_kde(v)
x_grid = np.linspace(v.min(), v.max(), 300)
plt.plot(x_grid, kde(x_grid), color="crimson", lw=2, label="Ядерная оценка плотности (KDE)")
plt.title("Оценка плотности распределения value")
plt.xlabel("Значение")
plt.ylabel("Плотность")
plt.legend()
plt.grid(alpha=0.5)
plt.tight_layout()
plt.savefig("density.png", dpi=150)
plt.show()

def pearson_test(v, dist, fit_kwargs=None, k=None):
    fit_kwargs = fit_kwargs or {}
    k = k or k_sturges
    params = dist.fit(v, **fit_kwargs)

    # Равновероятные интервалы по квантилям проверяемого распределения:
    # ожидаемая частота в каждом n/k, условие >= 5 выполняется автоматически
    edges = dist.ppf(np.linspace(0, 1, k + 1), *params)
    edges = np.unique(edges)          # защита от совпадающих границ
    edges[0], edges[-1] = -np.inf, np.inf

    obs, _ = np.histogram(v, bins=edges)
    exp = np.diff(dist.cdf(edges, *params)) * len(v)

    # Интервалы с ожидаемой частотой < 5 ОБЪЕДИНЯЕМ с соседними (не удаляем)
    o, e = [obs[0]], [exp[0]]
    for oi, ei in zip(obs[1:], exp[1:]):
        if ei < 5:
            o[-1] += oi
            e[-1] += ei
        else:
            o.append(oi)
            e.append(ei)
    o, e = np.array(o), np.array(e)
    k_used = len(o)

    # m — число оценённых по выборке параметров распределения
    m = len(params) - (1 if "floc" in fit_kwargs else 0)
    df = k_used - 1 - m
    if df < 1:
        return np.nan, np.nan, k_used, df, params
    chi2, p = stats.chisquare(o, e, ddof=m)   # ddof = m: поправка за оценённые параметры
    return chi2, p, k_used, df, params

# --- Проверка гипотезы о характере распределения (хи-квадрат Пирсона) ---
# Число интервалов по правилу Стёрджеса
k_sturges = int(np.ceil(1 + np.log2(len(v))))
candidates = {
    "нормальное":       (stats.norm,    {}),
    "логнормальное":    (stats.lognorm, {"floc": 0}),
    "экспоненциальное": (stats.expon,   {"floc": 0}),
}

print(f"Проверка гипотезы по критерию хи-квадрат "
      f"(k = {k_sturges} интервалов по правилу Стёрджеса):")
best_name, best_stat, best_p = None, np.inf, -np.inf
results = []
for name, (dist, kw) in candidates.items():
    chi2, p, k_used, dfree, params = pearson_test(v, dist, kw)
    results.append((name, chi2, p, k_used, dfree))
    print(f"  {name:20s} chi2 = {chi2:10.2f}, p-value = {p:.4f}, "
        f"df = {dfree} (интервалов использовано: {k_used})")
    if not np.isnan(p) and p > best_p:      # лучший кандидат — по максимальному p-value
        best_name, best_stat, best_p = name, chi2, p

# Вывод: если p-value > 0.05, гипотеза о распределении не отвергается
print(f"\nЛучший кандидат: {best_name} (chi2 = {best_stat:.2f}, p-value = {best_p:.4f})")
if best_p > 0.05:
    print(f"Вывод: на уровне значимости 0.05 гипотеза о {best_name} распределении НЕ отвергается.")
else:
    print(f"Вывод: на уровне значимости 0.05 гипотеза о {best_name} распределении ОТВЕРГАЕТСЯ, "
          "распределение значимо отличается от проверенных законов.")

# --- Интервальные оценки числовых характеристик ---
alpha = 0.05          # уровень значимости (доверительная вероятность 95%)
v = df["value"].dropna()
n = len(v)
mean = v.mean()
std = v.std(ddof=1)   # исправленное (выборочное) стандартное отклонение

print(f"Доверительная вероятность: {1 - alpha:.0%}, объём выборки: n = {n}\n")

# 1) Доверительный интервал для математического ожидания (критерий Стьюдента)
t_crit = stats.t.ppf(1 - alpha / 2, df=n - 1)   # критическое значение t
margin = t_crit * std / np.sqrt(n)
ci_mean = (mean - margin, mean + margin)
print(f"Среднее = {mean:.2f}")
print(f"  {1-alpha:.0%} доверительный интервал для среднего: "
      f"({ci_mean[0]:.2f}; {ci_mean[1]:.2f})")

# 2) Доверительный интервал для дисперсии и стандартного отклонения
chi_lower = stats.chi2.ppf(1 - alpha / 2, df=n - 1)   # левая критическая точка
chi_upper = stats.chi2.ppf(alpha / 2,     df=n - 1)   # правая критическая точка
ci_var  = ((n - 1) * v.var(ddof=1) / chi_lower,
           (n - 1) * v.var(ddof=1) / chi_upper)
ci_std  = (np.sqrt(ci_var[0]), np.sqrt(ci_var[1]))
print(f"Дисперсия = {v.var(ddof=1):.2f}")
print(f"  {1-alpha:.0%} доверительный интервал для дисперсии: "
      f"({ci_var[0]:.2f}; {ci_var[1]:.2f})")
print(f"Стандартное отклонение = {std:.2f}")
print(f"  {1-alpha:.0%} доверительный интервал для СКО: "
      f"({ci_std[0]:.2f}; {ci_std[1]:.2f})")

# 3) Доверительный интервал для медианы (бутстреп)
rng = np.random.default_rng(42)
boot_medians = np.median(rng.choice(v, size=(n, 5000), replace=True), axis=1)
ci_med = np.quantile(boot_medians, [alpha / 2, 1 - alpha / 2])
print(f"Медиана = {v.median():.2f}")
print(f"  {1-alpha:.0%} доверительный интервал для медианы (бутстреп): "
      f"({ci_med[0]:.2f}; {ci_med[1]:.2f})")

# ============================================================
# Оценка дополнительных данных, связанных с изучаемыми данными
# ============================================================

# --- 1. Удаление нерелевантных данных ---
print("=== 1. Очистка от нерелевантных данных ===")
n_before = len(df)

# Смотрим, какие флаги встречаются и сколько их
print("Распределение по флагам:")
print(df["flag_label"].value_counts(), "\n")

# Убираем строки с пометками о ненадёжных/служебных данных
# (обычно флаги вида 'Data missing', 'Not applicable', 'Estimate' и т.п.)
irrelevant_flags = ["Data missing", "Not applicable", "Not available"]
mask_flags = ~df["flag_label"].isin(irrelevant_flags)

# Убираем нулевые и отрицательные значения — физически бессмысленны
# для числа гостей, и записи, где страна не указана
mask_clean = (
    mask_flags
    & (df["value"] > 0)
    & df["reporter_area_label"].notna()
)

df_clean = df[mask_clean].copy()
print(f"Строк до очистки: {n_before}, после: {len(df_clean)} "
      f"(удалено {n_before - len(df_clean)})\n")

# --- 2. Сравнение со статистикой за другой период ---
print("=== 2. Сравнение периодов ===")
years = sorted(df_clean["year"].dropna().unique())
print(f"Доступные годы: {years}")

if len(years) >= 2:
    last_year, prev_year = years[-1], years[-2]
    s_last = df_clean[df_clean["year"] == last_year].set_index(
        "reporter_area_label")["value"]
    s_prev = df_clean[df_clean["year"] == prev_year].set_index(
        "reporter_area_label")["value"]

    common = s_last.index.intersection(s_prev.index)
    growth = (s_last[common] / s_prev[common] - 1) * 100

    comparison = pd.DataFrame({
        f"{prev_year}": s_prev[common],
        f"{last_year}": s_last[common],
        "Изменение, %": growth.round(1),
    }).sort_values(f"{last_year}", ascending=False)
    print(f"Сравнение {prev_year} и {last_year} (топ-10 стран):")
    print(comparison.head(10), "\n")

    print(f"Среднее значение в {prev_year}: {s_prev.mean():.2f}")
    print(f"Среднее значение в {last_year}: {s_last.mean():.2f}")
    print(f"Изменение среднего: {(s_last.mean() / s_prev.mean() - 1) * 100:+.1f}%\n")

    # Визуальное сравнение периодов
    top10 = comparison.head(10)
    x = np.arange(len(top10))
    plt.figure(figsize=(12, 6))
    plt.bar(x - 0.2, top10[prev_year_str := f"{prev_year}"], width=0.4,
            label=str(prev_year), color="lightsteelblue", edgecolor="black")
    plt.bar(x + 0.2, top10[f"{last_year}"], width=0.4,
            label=str(last_year), color="steelblue", edgecolor="black")
    plt.xticks(x, top10.index, rotation=45, ha="right")
    plt.title(f"Сравнение периодов {prev_year} и {last_year} (топ-10 стран)")
    plt.xlabel("Страна")
    plt.ylabel("Значение")
    plt.legend()
    plt.grid(axis="y", alpha=0.5)
    plt.tight_layout()
    plt.savefig("periods.png", dpi=150)
    plt.show()

# --- 3. Сравнение данных внутри выборки ---
print("=== 3. Сравнение внутри выборки ===")

# Группировка по показателям (indicator_label): среднее, максимум, разброс
by_indicator = df_clean.groupby("indicator_code")["value"].agg(
    ["mean", "median", "max", "std", "count"]
).sort_values("mean", ascending=False)

# Доля каждого показателя в общем объёме (концентрация)
total = df_clean["value"].sum()
share = df_clean.groupby("indicator_code")["value"].sum() / total * 100
top3_share = share.sort_values(ascending=False).head(3).sum()
print(f"На топ-3 показателя приходится {top3_share:.1f}% всех значений "
      f"(всего показателей: {len(share)})")

# Коэффициент вариации между показателями
print(f"Коэффициент вариации между показателями: "
      f"{by_indicator['mean'].std() / by_indicator['mean'].mean():.2%}\n")

# Boxplot по показателям: разброс внутри каждого показателя
# --- ИЗМЕНЕНО: по оси X теперь indicator_label вместо reporter_area_label ---
top_indicators = by_indicator.head(10).index
box_data = [df_clean.loc[df_clean["indicator_code"] == i, "value"]
            for i in top_indicators]

plt.figure(figsize=(12, 6))
plt.boxplot(box_data, tick_labels=top_indicators)
plt.title("Разброс значений по индикаторам")
plt.xlabel("Индикатор")
plt.ylabel("Значение")
plt.xticks(rotation=45, ha="right")
plt.grid(axis="y", alpha=0.5)
plt.tight_layout()
plt.savefig("boxplot.png", dpi=150)
plt.show()

import seaborn as sns

# Предполагаем, что df_clean уже подготовлен (очищен от нерелевантных данных)
# Если нет — вставьте сюда блок очистки из предыдущего ответа

# --- Скрипичная диаграмма по топ-10 странам ---
print("=== 4. Скрипичная диаграмма (violin plot) ===")

# Топ-10 стран по среднему значению
by_country = df_clean.groupby("indicator_code")["value"].agg(
    ["mean", "median", "max", "std", "count"]
).sort_values("mean", ascending=False)
top_countries = by_country.head(10).index

# Подготавливаем данные для seaborn: нужны два столбца — страна и значение
data_for_plot = df_clean[df_clean["indicator_code"].isin(top_countries)][[
    "indicator_code", "value"
]].copy()

plt.figure(figsize=(12, 6))
sns.violinplot(
    data=data_for_plot,
    x="indicator_code",
    y="value",
    hue="indicator_code",
    palette="Blues_d",
    inner="quartile",          # показывает медиану и квартили внутри скрипки
    linewidth=1.2,
    cut=0                      # обрезает «усы» скрипки по крайним точкам
)
plt.title("Распределение значений по индикаторам")
plt.xlabel("Индикатор")
plt.ylabel("Значение (число гостей)")
plt.xticks(rotation=45, ha="right")
plt.grid(axis="y", alpha=0.4)
plt.tight_layout()
plt.savefig("violin_plot.png", dpi=150)
plt.show()
