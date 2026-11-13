import pandas as pd

input_file = "DS_RP_TD_MENAGES_TPH_COMP_2023_data.csv"
output_file = "data_cut3.csv"

# Пробуем разные кодировки по очереди
encodings = ["utf-8", "cp1251", "latin-1", "iso-8859-1"]
df = None

for enc in encodings:
    try:
        df = pd.read_csv(input_file, encoding=enc, sep=";")
        print(f"Файл успешно прочитан с кодировкой: {enc}")
        break
    except UnicodeDecodeError:
        continue
    except Exception as e:
        print(f"Ошибка при чтении с {enc}: {e}")
        continue

if df is None:
    raise RuntimeError("Не удалось прочитать файл ни с одной из проверенных кодировок.")

# Вывод списка полей (названий колонок)
print("\nСписок полей в файле:")
for col in df.columns:
    print(col)

# Фильтрация: TIME_PERIOD == 2023, AGE == "_T", RP_MEASURE == "DWELLINGS",
# TPH в ["11", "MF22"] и GEO_OBJECT == "AAV2020"

# Маска для TIME_PERIOD
if pd.api.types.is_numeric_dtype(df["TIME_PERIOD"]):
    time_mask = df["TIME_PERIOD"] == 2023
else:
    time_mask = df["TIME_PERIOD"].astype(str).str.strip() == "2023"

# Маска для AGE == "_T"
age_mask = df["AGE"].astype(str).str.strip() == "_T"

# Маска для RP_MEASURE == "DWELLINGS"
measure_mask = df["RP_MEASURE"].astype(str).str.strip() == "DWELLINGS"

# Маска для TPH: "11" или "MF22" (с нормализацией пробелов)
tph_mask = df["TPH"].astype(str).str.strip().isin(["11", "MF22"])

# Маска для GEO_OBJECT == "AAV2020"
geo_mask = df["GEO_OBJECT"].astype(str).str.strip() == "AAV2020"

# Объединяем все условия
df_filtered = df[time_mask & age_mask & measure_mask & tph_mask & geo_mask]

df_filtered.to_csv(output_file, index=False)
print(f"Данные за 2023 год с AGE = '_T', RP_MEASURE = 'DWELLINGS' и TPH в ['11', 'MF22'] сохранены в файл {output_file}. Всего строк: {len(df_filtered)}")
