import pandas as pd


input_file = "DS_DECES_MORTALITE_SERIES_data.csv"
output_file = "data_cut1.csv"

# Пробуем разные кодировки
encodings = ["utf-8", "cp1251", "latin-1", "iso-8859-1"]
df = None

for enc in encodings:
    try:
        # sep=";" — потому что в файле разделитель точка с запятой
        df = pd.read_csv(input_file, encoding=enc, sep=";")
        print(f"Файл успешно прочитан с кодировкой: {enc} и разделителем ';'")
        break
    except UnicodeDecodeError:
        continue
    except Exception as e:
        print(f"Ошибка при чтении с {enc}: {e}")
        continue

if df is None:
    raise RuntimeError("Не удалось прочитать файл ни с одной из проверенных кодировок.")

# Вывод списка полей
print("\nСписок полей в файле:")
for col in df.columns:
    print(col)

# Автопоиск колонок (на случай лишних пробелов/символов)
time_col = geo_col = age_col = sex_col = ec_measure_col = None
for col in df.columns:
    if "TIME_PERIOD" in col and time_col is None:
        time_col = col
    if "GEO_OBJECT" in col and geo_col is None:
        geo_col = col
    if "AGE" in col and age_col is None:
        age_col = col
    if "SEX" in col and sex_col is None:
        sex_col = col
    if "EC_MEASURE" in col and ec_measure_col is None:
        ec_measure_col = col

# Проверка, что все нужные колонки найдены
missing = []
if time_col is None: missing.append("TIME_PERIOD")
if geo_col is None: missing.append("GEO_OBJECT")
if age_col is None: missing.append("AGE")
if sex_col is None: missing.append("SEX")
if ec_measure_col is None: missing.append("EC_MEASURE")

if missing:
    raise KeyError(f"В файле не найдены следующие колонки: {', '.join(missing)}. Проверьте список полей выше.")

print(f"\nИспользуемые колонки:")
print(f"  TIME_PERIOD:   '{time_col}'")
print(f"  GEO_OBJECT:    '{geo_col}'")
print(f"  AGE:           '{age_col}'")
print(f"  SEX:           '{sex_col}'")
print(f"  EC_MEASURE:   '{ec_measure_col}'")

# Фильтрация по TIME_PERIOD = 2019
if pd.api.types.is_numeric_dtype(df[time_col]):
    mask_time = df[time_col] == 2019
else:
    mask_time = df[time_col].astype(str).str.strip() == "2019"

# Исключение записей, где GEO_OBJECT = FRANCE (регистронезависимо + пробелы)
mask_geo = df[geo_col].astype(str).str.strip().str.upper() != "FRANCE"

# В AGE должно быть "_T"
mask_age = df[age_col].astype(str).str.contains("_T", na=False)

# В SEX должно быть "_T"
mask_sex = df[sex_col].astype(str).str.contains("_T", na=False)

# EC_MEASURE должно быть строго "DTH_PLACE_RES" (с учётом пробелов)
mask_ec = df[ec_measure_col].astype(str).str.strip() == "DTH_PLACE_RES"

# Применяем все условия
df_filtered = df[mask_time & mask_geo & mask_age & mask_sex & mask_ec]

# Вывод результатов
print(f"\nКоличество записей после всех фильтров: {len(df_filtered)}")
if not df_filtered.empty:
    print("\nПервые 5 строк отфильтрованных данных:")
    print(df_filtered.head())
else:
    print("\nЗаписей, удовлетворяющих всем условиям, не найдено.")

# Сохранение результата
df_filtered.to_csv(output_file, index=False, sep=";")
print(f"\nОтфильтрованные данные сохранены в файл: {output_file}")