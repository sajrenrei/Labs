import numpy as np


data = np.array([
    [72, 120, 80, 36.6, 98],
    [85, 135, 88, 37.1, 97],
    [68, 110, 72, 36.5, 99],
    [95, 145, 92, 37.4, 95],
    [76, 125, 82, 36.8, 98],
    [110, 155, 96, 38.0, 93],
    [64, 105, 68, 36.4, 99],
    [88, 130, 85, 37.0, 97],
    [102, 150, 94, 37.6, 94],
    [74, 118, 78, 36.7, 98]
])


patients_count = data.shape[0]
features_count = data.shape[1]
dimensions = data.ndim
total_elements = data.size

print("=== ЗАВДАННЯ 1 ===")
print(f"Кількість пацієнтів: {patients_count}")
print(f"Кількість показників: {features_count}")
print(f"Розмірність масиву: {dimensions}")
print(f"Кількість значень: {total_elements}\n")


hr = data[:, 0]
sbp = data[:, 1]
dbp = data[:, 2]
temperature = data[:, 3]
spo2 = data[:, 4]

first_patient = data[0]
last_patient = data[-1]
hr_p4 = data[3, 0]
first_5_rows = data[:5]

print("=== ЗАВДАННЯ 2 ===")
print("Усі значення HR:", hr)
print("Усі значення температури:", temperature)
print("Усі значення SpO2:", spo2)
print("Дані першого пацієнта:", first_patient)
print("Дані останнього пацієнта:", last_patient)
print("HR четвертого пацієнта:", hr_p4)
print("Перші 5 рядків масиву:\n", first_5_rows, "\n")


corrected_temperature = temperature - 0.2
pulse_pressure = sbp - dbp
hr_norm = (hr - hr.min()) / (hr.max() - hr.min())

print("=== ЗАВДАННЯ 3 ===")
print("Скоригована температура:\n", corrected_temperature)
print("Пульсовий тиск (PP):\n", pulse_pressure)
print("Нормалізовані значення HR (0..1):\n", hr_norm, "\n")


means = np.mean(data, axis=0)
medians = np.median(data, axis=0)
stds = np.std(data, axis=0)
vars_ = np.var(data, axis=0)
mins = np.min(data, axis=0)
maxs = np.max(data, axis=0)
p25 = np.percentile(data, 25, axis=0)
p75 = np.percentile(data, 75, axis=0)

feature_names = ["HR", "SBP", "DBP", "Temp", "SpO2"]

print("=== ЗАВДАННЯ 4 ===")
header = f"{'Показник':<10} {'Mean':<8} {'Median':<8} {'Std':<8} {'Min':<8} {'Max':<8} {'P25':<8} {'P75':<8}"
print(header)
print("-" * len(header))

for i in range(len(feature_names)):
    print(f"{feature_names[i]:<10} "
          f"{means[i]:<8.2f} "
          f"{medians[i]:<8.2f} "
          f"{stds[i]:<8.2f} "
          f"{mins[i]:<8.2f} "
          f"{maxs[i]:<8.2f} "
          f"{p25[i]:<8.2f} "
          f"{p75[i]:<8.2f}")
print()


mean_per_feature = np.mean(data, axis=0)
mean_per_patient = np.mean(data, axis=1)
min_per_feature = np.min(data, axis=0)
max_per_feature = np.max(data, axis=0)

print("=== ЗАВДАННЯ 5 ===")
print("Середнє значення кожного показника (axis=0):\n", mean_per_feature)
print("\nСереднє значення всіх показників кожного пацієнта (axis=1):\n", mean_per_patient)
print("\nМінімальне значення кожного показника (axis=0):\n", min_per_feature)
print("\nМаксимальне значення кожного показника (axis=0):\n", max_per_feature, "\n")


abnormal_hr = (hr < 60) | (hr > 100)
abnormal_sbp = (sbp < 90) | (sbp > 140)
abnormal_dbp = (dbp < 60) | (dbp > 90)
abnormal_temp = (temperature < 36.0) | (temperature > 37.5)
abnormal_spo2 = (spo2 < 95) | (spo2 > 100)

print("=== ЗАВДАННЯ 6 ===")
print("Аномальний HR (значення):", hr[abnormal_hr])
print("Номери пацієнтів (індекси):", np.where(abnormal_hr)[0])
print("Аномальний SBP (значення):", sbp[abnormal_sbp])
print("Номери пацієнтів (індекси):", np.where(abnormal_sbp)[0])
print("Аномальний DBP (значення):", dbp[abnormal_dbp])
print("Номери пацієнтів (індекси):", np.where(abnormal_dbp)[0])
print("Аномальна температура (значення):", temperature[abnormal_temp])
print("Номери пацієнтів (індекси):", np.where(abnormal_temp)[0])
print("Аномальний SpO2 (значення):", spo2[abnormal_spo2])
print("Номери пацієнтів (індекси):", np.where(abnormal_spo2)[0], "\n")


def analyze_patient(patient_data):
    p_hr, p_sbp, p_dbp, p_temp, p_spo2 = patient_data
    pp = p_sbp - p_dbp

    is_abnormal = (
            (p_hr < 60 or p_hr > 100) or
            (p_sbp < 90 or p_sbp > 140) or
            (p_dbp < 60 or p_dbp > 90) or
            (p_temp < 36.0 or p_temp > 37.5) or
            (p_spo2 < 95 or p_spo2 > 100)
    )

    return {
        "HR": p_hr,
        "SBP": p_sbp,
        "DBP": p_dbp,
        "Temperature": p_temp,
        "SpO2": p_spo2,
        "Pulse_Pressure": pp,
        "Has_Abnormalities": is_abnormal
    }


print("=== ЗАВДАННЯ 7 ===")
print(analyze_patient(data[0]), "\n")

print("=== ЗАВДАННЯ 8 ===")
print("==========================================")
print("  АНАЛІЗ ФІЗІОЛОГІЧНИХ ВИМІРЮВАНЬ")
print("==========================================")
print(f"\nКількість пацієнтів: {data.shape[0]}\n")
print("СЕРЕДНІ ЗНАЧЕННЯ:")
print(f"HR:   {np.mean(hr):>5.1f} уд/хв")
print(f"SBP:  {np.mean(sbp):>5.1f} мм рт. ст.")
print(f"DBP:  {np.mean(dbp):>5.1f} мм рт. ст.")
print(f"Temp: {np.mean(temperature):>5.1f} °C")
print(f"SpO2: {np.mean(spo2):>5.1f} %\n")
print("АНОМАЛЬНІ ЗНАЧЕННЯ:")

masks = [abnormal_hr, abnormal_sbp, abnormal_dbp, abnormal_temp, abnormal_spo2]
values_list = [hr, sbp, dbp, temperature, spo2]
units = ["уд/хв", "мм рт. ст.", "мм рт. ст.", "°C", "%"]

for name, mask, vals, unit in zip(feature_names, masks, values_list, units):
    indices = np.where(mask)[0]
    if len(indices) > 0:
        print(f"\n{name}:")
        for idx in indices:
            val = vals[idx]
            val_str = f"{val:.1f}" if isinstance(val, float) and not val.is_integer() else f"{int(val)}"
            print(f"P{idx + 1:02d} — {val_str} {unit}")
