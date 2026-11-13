import pandas as pd

input_file = "DS_IPC_PRINC_data.csv"
output_file = "data_cut2.csv"

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

# --- Маски для фильтрации ---

# TIME_PERIOD == 2019
if pd.api.types.is_numeric_dtype(df["TIME_PERIOD"]):
    mask_time = df["TIME_PERIOD"] == 2019
else:
    mask_time = df["TIME_PERIOD"].astype(str).str.strip() == "2019"

# BASE_PER == 2025
if "BASE_PER" not in df.columns:
    raise KeyError("В файле отсутствует колонка BASE_PER")
if pd.api.types.is_numeric_dtype(df["BASE_PER"]):
    mask_base = df["BASE_PER"] == 2025
else:
    mask_base = df["BASE_PER"].astype(str).str.strip() == "2025"

# GEO_OBJECT != "FRANCE"
if "GEO_OBJECT" not in df.columns:
    raise KeyError("В файле отсутствует колонка GEO_OBJECT")
mask_geo = df["GEO_OBJECT"].astype(str).str.strip() != "FRANCE"

# TPH_CPI == "_T"
if "TPH_CPI" not in df.columns:
    raise KeyError("В файле отсутствует колонка TPH_CPI")
mask_tph = df["TPH_CPI"].astype(str).str.strip() == "_T"

# COICOP_2018 == "_Z"
if "COICOP_2018" not in df.columns:
    raise KeyError("В файле отсутствует колонка COICOP_2018")
mask_coicop = df["COICOP_2018"].astype(str).str.strip() == "_Z"

# IND_TYPE == "Y_VAR"
if "IND_TYPE" not in df.columns:
    raise KeyError("В файле отсутствует колонка IND_TYPE")
if pd.api.types.is_numeric_dtype(df["IND_TYPE"]):
    # Если вдруг это число — сравниваем как число (но для "Y_VAR" это маловероятно)
    mask_ind = df["IND_TYPE"] == "Y_VAR"  # сработает как False для чисел, что корректно
else:
    mask_ind = df["IND_TYPE"].astype(str).str.strip() == "Y_VAR"

# Применяем все условия одновременно
df_filtered = df[mask_time & mask_base & mask_geo & mask_tph & mask_coicop & mask_ind]

df_filtered.to_csv(output_file, index=False)
print(f"Данные отфильтрованы и сохранены в файл {output_file}.")
print(f"Всего строк после фильтрации: {len(df_filtered)}")