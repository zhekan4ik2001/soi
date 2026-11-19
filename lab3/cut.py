import csv

# Коды стран, которые оставляем
ALLOWED = {"RUS"}

with open("data.csv", "r", encoding="utf-8", newline="") as src, \
     open("data_cutting.csv", "w", encoding="utf-8", newline="") as dst:

    reader = csv.DictReader(src)
    writer = csv.DictWriter(dst, fieldnames=reader.fieldnames)
    writer.writeheader()

    kept = 0
    for row in reader:
        if row["geoUnit"] in ALLOWED:
            writer.writerow(row)
            kept += 1

print(f"Готово: в data_cutting.csv записано {kept} строк")
