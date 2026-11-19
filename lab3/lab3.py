"""
Практическое задание №3. Исследование временных рядов.

Данные: data_cutting.csv
Ряды: CHN (Китай), RUS (Россия), USA (США), 2000–2025 гг.

Установка библиотек:
    pip install pandas numpy matplotlib scipy statsmodels

Запуск:
    python analysis_time_series.py

Результаты:
    - подробные вычисления выводятся в консоль;
    - графики сохраняются в папку results_time_series.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import linregress, t as student_t
from statsmodels.tsa.stattools import acf, adfuller, kpss
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import acorr_ljungbox

# --------------------------- НАСТРОЙКИ -------------------------------------

DATA_FILE = "data_cutting.csv"
OUTPUT_DIR = Path("results_time_series")
COUNTRIES = { "RUS": "Россия" }
ALPHA = 0.05
MOVING_WINDOW = 3
MAX_ACF_LAG = 10

OUTPUT_DIR.mkdir(exist_ok=True)


# --------------------------- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ------------------------

def print_section(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def print_test(name, statistic, p_value, h0_text):
    decision = "H0 НЕ отвергается" if p_value >= ALPHA else "H0 отвергается"
    print(f"{name}: статистика = {statistic:.5f}; p-value = {p_value:.5f}")
    print(f"  Решение при alpha={ALPHA}: {decision}. H0: {h0_text}")


def save_figure(fig, filename):
    path = OUTPUT_DIR / filename
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"График сохранён: {path.resolve()}")


def moving_average_with_edges(y, window=3):
    """
    Простая центрированная скользящая средняя.
    Для крайних элементов используется линейное продолжение,
    рассчитанное по среднему приросту на последнем активном участке.
    """
    y = np.asarray(y, dtype=float)
    if window < 2 or window > len(y):
        raise ValueError("Некорректная ширина окна скользящей средней.")

    core = pd.Series(y).rolling(window=window, center=True).mean().to_numpy()
    valid = np.flatnonzero(~np.isnan(core))
    first, last = valid[0], valid[-1]

    # Средний абсолютный прирост на активном участке
    avg_increment = (core[last] - core[first]) / (last - first)
    result = core.copy()

    for i in range(first - 1, -1, -1):
        result[i] = result[i + 1] - avg_increment
    for i in range(last + 1, len(y)):
        result[i] = result[i - 1] + avg_increment

    return result


def irwin_criterion(y):
    """Критерий Ирвина: lambda_i = |y_i-y_(i-1)| / s."""
    y = np.asarray(y, dtype=float)
    s = np.std(y, ddof=1)
    if s == 0:
        return np.zeros(len(y) - 1)
    return np.abs(np.diff(y)) / s


def foster_stuart_test(y):
    """
    Критерий Фостера–Стюарта для выявления тенденции.
    u_i=1, если уровень больше всех предыдущих;
    d_i=1, если уровень меньше всех предыдущих.
    """
    y = np.asarray(y, dtype=float)
    highs, lows = [], []
    for i in range(1, len(y)):
        highs.append(int(y[i] > np.max(y[:i])))
        lows.append(int(y[i] < np.min(y[:i])))

    S = sum(highs) + sum(lows)
    D = sum(highs) - sum(lows)
    n = len(y)

    # Приближённые математическое ожидание и дисперсия суммы S
    # и статистика по разности D (классическая форма критерия).
    mean_S = sum(2 / (i + 1) for i in range(1, n))
    var_S = sum(2 / (i + 1) - 4 / (i + 1) ** 2 for i in range(1, n))
    z_S = (S - mean_S) / np.sqrt(var_S) if var_S > 0 else np.nan

    # Для D используется приближённая нормировка через сумму дисперсий
    # индикаторов экстремумов; результат рассматривается как диагностический.
    var_D = var_S
    z_D = D / np.sqrt(var_D) if var_D > 0 else np.nan
    return S, D, z_S, z_D


def mean_comparison_trend_test(y):
    """Сравнение средних первой и второй частей ряда (Welch t-test)."""
    y = np.asarray(y, dtype=float)
    middle = len(y) // 2
    first, second = y[:middle], y[middle:]
    result = stats.ttest_ind(first, second, equal_var=False)
    return first, second, result


def runs_test(residuals):
    """Критерий серий относительно медианы остатков."""
    r = np.asarray(residuals, dtype=float)
    med = np.median(r)
    signs = np.where(r > med, 1, np.where(r < med, -1, 0))
    signs = signs[signs != 0]

    if len(signs) < 2:
        return np.nan, np.nan, np.nan, 0, 0

    runs = 1 + np.sum(signs[1:] != signs[:-1])
    n_plus = np.sum(signs == 1)
    n_minus = np.sum(signs == -1)
    n = n_plus + n_minus

    expected = 1 + 2 * n_plus * n_minus / n
    variance = (
        2 * n_plus * n_minus * (2 * n_plus * n_minus - n)
        / (n ** 2 * (n - 1))
    ) if n > 1 else np.nan
    z = (runs - expected) / np.sqrt(variance) if variance > 0 else np.nan
    max_run = 1
    current = 1
    for i in range(1, len(signs)):
        if signs[i] == signs[i - 1]:
            current += 1
            max_run = max(max_run, current)
        else:
            current = 1

    return runs, expected, z, max_run, n


def turning_points_test(residuals):
    """Критерий поворотных точек из теоретических материалов."""
    r = np.asarray(residuals, dtype=float)
    n = len(r)
    p = sum(
        (r[i] > r[i - 1] and r[i] > r[i + 1]) or
        (r[i] < r[i - 1] and r[i] < r[i + 1])
        for i in range(1, n - 1)
    )
    mean_p = 2 * (n - 2) / 3
    var_p = (16 * n - 29) / 90
    sd_p = np.sqrt(var_p)
    z = (p - mean_p) / sd_p
    return p, mean_p, sd_p, z


def residual_normality_tests(residuals):
    """Проверки нормальности: Шапиро–Уилк и оценки асимметрии/эксцесса."""
    r = np.asarray(residuals, dtype=float)
    shapiro = stats.shapiro(r)
    skew = stats.skew(r, bias=False)
    excess = stats.kurtosis(r, fisher=True, bias=False)
    se_skew = np.sqrt(6 / len(r))
    se_excess = np.sqrt(24 / len(r))
    return shapiro, skew, excess, se_skew, se_excess


def fit_polynomial_models(t, y):
    """Подбор полиномиальных трендов 1–3 порядка методом МНК."""
    models = []
    for degree in (1, 2, 3):
        coef = np.polyfit(t, y, degree)
        fitted = np.polyval(coef, t)
        residuals = y - fitted
        sse = np.sum(residuals ** 2)
        sst = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - sse / sst if sst else np.nan
        k = degree + 1
        n = len(y)
        aic = n * np.log(sse / n) + 2 * k if sse > 0 else -np.inf
        bic = n * np.log(sse / n) + k * np.log(n) if sse > 0 else -np.inf
        models.append({
            "degree": degree, "coef": coef, "fitted": fitted,
            "residuals": residuals, "sse": sse, "r2": r2,
            "aic": aic, "bic": bic,
        })
    return models


# --------------------------- ЗАГРУЗКА ДАННЫХ -------------------------------

df = pd.read_csv(DATA_FILE)

required = {"geoUnit", "year", "value"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"В CSV отсутствуют обязательные столбцы: {missing}")

df["year"] = pd.to_numeric(df["year"], errors="coerce")
df["value"] = pd.to_numeric(df["value"], errors="coerce")
df = df.dropna(subset=["geoUnit", "year", "value"])

print_section("ОБЩАЯ ИНФОРМАЦИЯ О ДАННЫХ")
print(f"Файл: {DATA_FILE}")
print(f"Количество наблюдений: {len(df)}")
print(f"Показатель: {df['indicatorId'].iloc[0] if 'indicatorId' in df.columns else 'не указан'}")
print(f"Период: {int(df.year.min())}–{int(df.year.max())}")
print("Количество наблюдений по странам:")
print(df.groupby("geoUnit").size().to_string())


# --------------------------- АНАЛИЗ КАЖДОЙ СТРАНЫ ---------------------------

for code, country_name in COUNTRIES.items():
    part = df[df.geoUnit == code].sort_values("year").copy()
    if part.empty:
        print(f"\nНет данных для {country_name} ({code}), пропуск.")
        continue

    years = part.year.to_numpy(dtype=int)
    y = part.value.to_numpy(dtype=float)
    n = len(y)
    t = np.arange(n)

    print_section(f"{country_name.upper()} ({code}): ПОЛНЫЙ АНАЛИЗ")
    print(f"Число уровней: n={n}; период: {years[0]}–{years[-1]}")
    print("\nИсходные значения:")
    print(part[["year", "value"]].to_string(index=False))

    # 1. Исходный график каждой страны
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(years, y, marker="o", linewidth=1.8, label="Исходный ряд")
    ax.set_title(f"{country_name}: исходный временной ряд")
    ax.set_xlabel("Год")
    ax.set_ylabel("Годовой прирост ВВП, %")
    ax.grid(True, alpha=0.3)
    ax.legend()
    save_figure(fig, f"{code}_01_original_series.png")

    # 2. Сериальные корреляции и коррелограмма
    print("\n--- СЕРИАЛЬНЫЕ КОРРЕЛЯЦИИ И КОРРЕЛОГРАММА ---")
    max_lag = min(MAX_ACF_LAG, n - 1)
    correlations = acf(y, nlags=max_lag, fft=False, adjusted=False)
    confidence = 1.96 / np.sqrt(n)
    print(f"Ориентировочные 95%-ные границы: ±{confidence:.4f}")
    for lag in range(1, max_lag + 1):
        state = "значима ориентировочно" if abs(correlations[lag]) > confidence else "в пределах границ"
        print(f"Лаг {lag:2d}: r={correlations[lag]: .5f} — {state}")

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(range(1, max_lag + 1), correlations[1:max_lag + 1])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(confidence, color="red", linestyle="--", label="95%-ные границы")
    ax.axhline(-confidence, color="red", linestyle="--")
    ax.set_title(f"{country_name}: коррелограмма исходного ряда")
    ax.set_xlabel("Лаг (лет)")
    ax.set_ylabel("Коэффициент автокорреляции")
    ax.set_xticks(range(1, max_lag + 1))
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    save_figure(fig, f"{code}_02_acf_original.png")

    # 3. Стационарность
    print("\n--- ПРОВЕРКА СТАЦИОНАРНОСТИ ---")
    print("ADF: H0 — ряд имеет единичный корень (нестационарен).")
    try:
        adf = adfuller(y, autolag="AIC")
        print_test("ADF", adf[0], adf[1], "ряд имеет единичный корень")
        print(f"  Критические значения: {adf[4]}")
    except Exception as exc:
        print(f"ADF не рассчитан: {exc}")

    print("KPSS: H0 — ряд стационарен относительно константы.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kpss_result = kpss(y, regression="c", nlags="auto")
        print_test("KPSS", kpss_result[0], kpss_result[1], "ряд стационарен")
        print(f"  Критические значения: {kpss_result[3]}")
    except Exception as exc:
        print(f"KPSS не рассчитан: {exc}")

    # 4. Сезонность и цикличность
    print("\n--- СЕЗОННАЯ И ЦИКЛИЧЕСКАЯ СОСТАВЛЯЮЩИЕ ---")
    print("Данные годовые. Внутригодовая сезонность (кварталы/месяцы)")
    print("по этому набору определить нельзя: в каждом году только одно наблюдение.")
    print("Циклические колебания рассматриваются по отклонениям от тренда;")
    print("их наличие само по себе не доказывает регулярный период цикла.")

    # 5. Аномалии: метод Ирвина
    print("\n--- АНОМАЛЬНЫЕ УРОВНИ: МЕТОД ИРВИНА ---")
    lambdas = irwin_criterion(y)
    print("lambda_i = |y_i - y_(i-1)| / s, где s — выборочное стандартное отклонение.")
    print("Табличный порог зависит от n и alpha; ниже выводятся расчётные значения.")
    print("Для предварительного поиска резких скачков используется ориентир lambda > 1.5,")
    print("но окончательное решение следует сверять с таблицей критических значений.")
    for i, value in enumerate(lambdas, start=1):
        mark = "  <-- потенциально аномальный скачок" if value > 1.5 else ""
        print(f"{years[i-1]}->{years[i]}: lambda={value:.4f}{mark}")

    # 6. Проверка наличия тренда: сравнение средних
    print("\n--- НАЛИЧИЕ ТРЕНДА: СРАВНЕНИЕ СРЕДНИХ ---")
    first, second, mean_test = mean_comparison_trend_test(y)
    print(f"Первая часть: n={len(first)}, среднее={np.mean(first):.5f}, "
          f"дисперсия={np.var(first, ddof=1):.5f}")
    print(f"Вторая часть: n={len(second)}, среднее={np.mean(second):.5f}, "
          f"дисперсия={np.var(second, ddof=1):.5f}")
    print_test("Welch t-test", mean_test.statistic, mean_test.pvalue,
               "средние двух частей равны")

    # 7. Критерий Фостера–Стюарта
    print("\n--- НАЛИЧИЕ ТРЕНДА: ФОСТЕР–СТЮАРТ ---")
    S, D, zS, zD = foster_stuart_test(y)
    print(f"S = {S}; D = {D}; стандартизированная статистика S = {zS:.5f}; "
          f"статистика D = {zD:.5f}")
    print("Сравнение |z| с 1.96 даёт приближённую проверку на уровне 5%.")
    print(f"Для S: {'сигнал тенденции' if abs(zS) > 1.96 else 'значимый сигнал не выявлен'}.")
    print("Статистика приведена как приближённая реализация критерия; для строгого")
    print("табличного вывода следует использовать таблицу/вариант формулы из курса.")

    # 8. Сглаживание скользящей средней
    print("\n--- СГЛАЖИВАНИЕ СКОЛЬЗЯЩЕЙ СРЕДНЕЙ ---")
    ma = moving_average_with_edges(y, MOVING_WINDOW)
    print(f"Ширина окна: {MOVING_WINDOW}")
    print(pd.DataFrame({"Год": years, "Исходный ряд": y,
                        "Скользящая средняя": ma}).to_string(index=False))

    # 9. Выбор полиномиального тренда и оценка МНК
    print("\n--- МОДЕЛИ ТРЕНДА И ОЦЕНКА ПАРАМЕТРОВ МНК ---")
    models = fit_polynomial_models(t, y)
    model_table = pd.DataFrame([
        {"Порядок": m["degree"], "R2": m["r2"], "AIC": m["aic"], "BIC": m["bic"]}
        for m in models
    ])
    print(model_table.to_string(index=False, float_format=lambda x: f"{x:.5f}"))

    # Выбираем модель с минимальным BIC; она не обязательно линейная.
    best = min(models, key=lambda m: m["bic"])
    fitted = best["fitted"]
    residuals = best["residuals"]
    degree = best["degree"]
    print(f"\nВыбрана полиномиальная модель порядка {degree} по минимуму BIC.")
    print("Коэффициенты полинома (от старшей степени к свободному члену):")
    print(np.array2string(best["coef"], precision=7))
    print(f"R² = {best['r2']:.5f}; SSE = {best['sse']:.5f}; "
          f"AIC = {best['aic']:.5f}; BIC = {best['bic']:.5f}")

    # F-критерий общей значимости регрессии
    k = degree + 1
    df_reg = degree
    df_res = n - k
    sst = np.sum((y - np.mean(y)) ** 2)
    ssr = sst - best["sse"]
    f_stat = (ssr / df_reg) / (best["sse"] / df_res) if df_res > 0 and best["sse"] > 0 else np.nan
    f_p = stats.f.sf(f_stat, df_reg, df_res) if np.isfinite(f_stat) else np.nan
    print_test("F-критерий модели", f_stat, f_p, "все коэффициенты при t равны нулю")

    # Сравнение исходного ряда, MA и модели тренда
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(years, y, marker="o", label="Исходный ряд")
    ax.plot(years, ma, linewidth=2, label=f"Скользящая средняя ({MOVING_WINDOW})")
    ax.plot(years, fitted, "--", linewidth=2, label=f"Полиномиальный тренд, порядок {degree}")
    ax.set_title(f"{country_name}: сглаживание и выбранный тренд")
    ax.set_xlabel("Год")
    ax.set_ylabel("Годовой прирост ВВП, %")
    ax.grid(True, alpha=0.3)
    ax.legend()
    save_figure(fig, f"{code}_03_trend_and_moving_average.png")

    # 10. Циклическая составляющая как остатки от тренда
    print("\n--- ЦИКЛИЧЕСКИЕ КОЛЕБАНИЯ ПОСЛЕ УДАЛЕНИЯ ТРЕНДА ---")
    print(f"Стандартное отклонение отклонений от тренда: {np.std(residuals, ddof=1):.5f}")
    turning = turning_points_test(residuals)
    print(f"Поворотные точки: p={turning[0]}, математическое ожидание={turning[1]:.3f}, "
          f"SD={turning[2]:.3f}, z={turning[3]:.3f}")
    print("Повторяющиеся отклонения могут указывать на цикличность, однако")
    print("для подтверждения периода цикла требуется более длинный ряд.")

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.axhline(0, color="black", linewidth=0.8)
    ax.plot(years, residuals, marker="o", label="Отклонения от тренда")
    ax.set_title(f"{country_name}: остатки трендовой модели")
    ax.set_xlabel("Год")
    ax.set_ylabel("Остаток, процентные пункты")
    ax.grid(True, alpha=0.3)
    ax.legend()
    save_figure(fig, f"{code}_04_detrended_residuals.png")

    # 11. Адекватность тренда и остаточная компонента
    print("\n--- ОЦЕНКА АДЕКВАТНОСТИ ТРЕНДА И АНАЛИЗ ОСТАТКОВ ---")
    print(f"Среднее остатков = {np.mean(residuals):.8f}")
    print(f"Стандартное отклонение остатков = {np.std(residuals, ddof=1):.5f}")
    print(f"RMSE = {np.sqrt(np.mean(residuals**2)):.5f}")
    print(f"MAE = {np.mean(np.abs(residuals)):.5f}")

    # Случайность: критерий серий и поворотных точек
    runs, exp_runs, z_runs, max_run, n_signs = runs_test(residuals)
    print("\nКритерий серий относительно медианы:")
    print(f"Количество серий={runs}; ожидаемое число={exp_runs:.3f}; "
          f"z={z_runs:.4f}; максимальная длина серии={max_run}")
    if np.isfinite(z_runs):
        print("Вывод: " + ("случайность отвергается (|z|>1.96)." if abs(z_runs) > 1.96
                           else "нет оснований отвергать случайность по критерию серий."))

    p_turn, mean_turn, sd_turn, z_turn = turning
    print("\nКритерий поворотных точек:")
    print(f"p={p_turn}; E(p)={mean_turn:.3f}; SD(p)={sd_turn:.3f}; z={z_turn:.4f}")
    print("Вывод: " + ("случайность отвергается." if abs(z_turn) > 1.96
                       else "нет оснований отвергать случайность по этому критерию."))

    # Нормальность
    print("\nПроверка нормальности:")
    shapiro, skew, excess, se_skew, se_excess = residual_normality_tests(residuals)
    print_test("Шапиро–Уилк", shapiro.statistic, shapiro.pvalue,
               "остатки имеют нормальное распределение")
    print(f"Асимметрия={skew:.5f}; стандартная ошибка≈{se_skew:.5f}; "
          f"|A|/SE={abs(skew)/se_skew:.3f}")
    print(f"Эксцесс={excess:.5f}; стандартная ошибка≈{se_excess:.5f}; "
          f"|E|/SE={abs(excess)/se_excess:.3f}")
    print("Приближённо большие отношения |A|/SE или |E|/SE (более 1.96)")
    print("свидетельствуют об отклонении от нормальности.")

    # Нулевое математическое ожидание
    t_stat, t_p = stats.ttest_1samp(residuals, popmean=0)
    print("\nПроверка среднего остатка:")
    print_test("Одновыборочный t-тест", t_stat, t_p,
               "математическое ожидание остатков равно нулю")

    # Независимость: Durbin-Watson и Ljung-Box
    dw = durbin_watson(residuals)
    print("\nПроверка независимости остатков:")
    print(f"Статистика Дарбина–Уотсона d={dw:.5f}")
    print("Ориентир: d около 2 — нет выраженной автокорреляции первого порядка;")
    print("d существенно меньше 2 — возможна положительная, больше 2 — отрицательная.")
    try:
        lb = acorr_ljungbox(residuals, lags=[min(5, n // 3)], return_df=True)
        print("Тест Льюнга–Бокса:")
        print(lb.to_string())
        lb_p = float(lb["lb_pvalue"].iloc[0])
        print("Вывод: " + ("автокорреляция остатков не выявлена."
                           if lb_p >= ALPHA else
                           "есть свидетельства автокорреляции остатков."))
    except Exception as exc:
        print(f"Тест Льюнга–Бокса не рассчитан: {exc}")

    # Графики остатков и Q-Q
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].plot(years, residuals, marker="o")
    axes[0].set_title("Остатки во времени")
    axes[0].set_xlabel("Год")
    axes[0].set_ylabel("Остаток")
    axes[0].grid(True, alpha=0.3)

    stats.probplot(residuals, dist="norm", plot=axes[1])
    axes[1].set_title("Q-Q график остатков")
    axes[1].grid(True, alpha=0.3)
    fig.suptitle(f"{country_name}: диагностика остаточной компоненты")
    save_figure(fig, f"{code}_05_residual_diagnostics.png")

    # Коррелограмма остатков
    residual_acf = acf(residuals, nlags=min(MAX_ACF_LAG, n - 1), fft=False)
    fig, ax = plt.subplots(figsize=(9, 4))
    lags = range(1, len(residual_acf))
    ax.bar(list(lags), residual_acf[1:])
    conf = 1.96 / np.sqrt(n)
    ax.axhline(conf, color="red", linestyle="--")
    ax.axhline(-conf, color="red", linestyle="--")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(f"{country_name}: коррелограмма остатков")
    ax.set_xlabel("Лаг")
    ax.set_ylabel("ACF")
    ax.grid(True, axis="y", alpha=0.3)
    save_figure(fig, f"{code}_06_residual_acf.png")

    # Сохранение расчётной таблицы для каждой страны
    result_table = pd.DataFrame({
        "year": years,
        "actual": y,
        "moving_average": ma,
        "trend": fitted,
        "residual": residuals,
    })
    result_table.to_csv(OUTPUT_DIR / f"{code}_calculations.csv",
                        index=False, encoding="utf-8-sig")
    print(f"\nТаблица расчётов сохранена: "
          f"{(OUTPUT_DIR / f'{code}_calculations.csv').resolve()}")


print_section("АНАЛИЗ ЗАВЕРШЁН")
print(f"Все графики и таблицы находятся в папке: {OUTPUT_DIR.resolve()}")
print("Интерпретируйте тесты с учётом небольшой длины рядов (26 наблюдений).")
