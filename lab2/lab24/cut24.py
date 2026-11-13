import pandas as pd

input_file = "DD_CNA_ECH_EXT_data.csv"
output_file = "data_cut4.csv"

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

# Фильтрация по TIME_PERIOD = 2019 и PRICES == V
# Сначала проверим тип данных в колонке TIME_PERIOD
if pd.api.types.is_numeric_dtype(df["TIME_PERIOD"]):
    time_filter = df["TIME_PERIOD"] == 2019
else:
    # Если это строки — пробуем и "2019", и с возможными пробелами
    time_filter = df["TIME_PERIOD"].astype(str).str.strip() == "2019"

# Проверка для PRICES == V (учитываем возможные пробелы и регистр)
prices_filter = df["PRICES"].astype(str).str.strip().str.upper() == "V"

df_filtered = df[time_filter & prices_filter]

df_filtered.to_csv(output_file, index=False)
print(f"Данные за 2019 год с PRICES == V сохранены в файл {output_file}. Всего строк: {len(df_filtered)}")
