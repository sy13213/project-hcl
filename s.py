
import numpy as np
import pandas as pd
import os

# ================= PROJECT CONFIGURATION =================

os.makedirs("data", exist_ok=True)
os.makedirs("reports", exist_ok=True)

FILE = "data/renewable_energy_data.csv"

SOURCES = ["Solar", "Wind", "Hydro", "Biomass", "Geothermal"]
REGIONS = ["North", "South", "East", "West", "Central"]

COLUMNS = [
    "record_id", "date", "region", "energy_source",
    "capacity_mw", "production_mwh",
    "capacity_factor_pct", "co2_avoided_tonnes"
]


# ================= 1. DATASET CREATION =================

def create_dataset():
    if os.path.exists(FILE):
        return

    records = []
    capacities = [120, 150, 200, 75, 60]
    factors = [0.19, 0.34, 0.46, 0.62, 0.74]
    co2_rates = [0.88, 0.82, 0.95, 0.78, 0.91]

    for i, source in enumerate(SOURCES):
        for month in range(1, 13):
            factor = factors[i] * (0.85 + month * 0.01)
            production = capacities[i] * 720 * factor

            records.append([
                len(records) + 1,
                f"2025-{month:02d}-01",
                REGIONS[(i + month) % 5],
                source,
                capacities[i],
                round(production, 2),
                round(factor * 100, 2),
                round(production * co2_rates[i], 2)
            ])

    pd.DataFrame(records, columns=COLUMNS).to_csv(FILE, index=False)
    print("60-record dataset created successfully!")


# ================= 2. FILE HANDLING =================

def load_data():
    df = pd.read_csv(FILE)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    numeric = [
        "record_id", "capacity_mw", "production_mwh",
        "capacity_factor_pct", "co2_avoided_tonnes"
    ]

    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())

    df["region"] = df["region"].fillna("Unknown")
    df["energy_source"] = df["energy_source"].fillna("Unknown")
    return df


def save_data(df):
    df.to_csv(FILE, index=False)


# ================= 3. CRUD OPERATIONS =================

def display_records():
    print(load_data().to_string(index=False))


def add_record():
    df = load_data()

    record = {
        "record_id": int(input("Record ID: ")),
        "date": input("Date (YYYY-MM-DD): "),
        "region": input("Region: ").title(),
        "energy_source": input("Energy Source: ").title(),
        "capacity_mw": float(input("Capacity (MW): ")),
        "production_mwh": float(input("Production (MWh): ")),
        "capacity_factor_pct": float(input("Capacity Factor (%): ")),
        "co2_avoided_tonnes": float(input("CO2 Avoided (Tonnes): "))
    }

    if record["record_id"] in df["record_id"].values:
        print("Record ID already exists!")
        return

    if any(record[k] < 0 for k in [
        "capacity_mw", "production_mwh",
        "capacity_factor_pct", "co2_avoided_tonnes"
    ]) or record["capacity_factor_pct"] > 100:
        print("Invalid numeric values!")
        return

    record["date"] = pd.to_datetime(record["date"], errors="raise")
    save_data(pd.concat([df, pd.DataFrame([record])], ignore_index=True))
    print("Record added successfully!")


def search_record():
    df = load_data()
    keyword = input("Enter ID, region or energy source: ").strip()

    result = df[
        df.astype(str).apply(
            lambda col: col.str.contains(keyword, case=False, na=False)
        ).any(axis=1)
    ]

    print(result.to_string(index=False) if not result.empty else "No record found.")


def update_record():
    df = load_data()
    rid = int(input("Enter Record ID: "))

    if rid not in df["record_id"].values:
        print("Record not found!")
        return

    field = input("Field to update: ").strip()
    if field not in COLUMNS[1:]:
        print("Invalid field!")
        return

    value = input("New value: ")

    if field == "date":
        value = pd.to_datetime(value, errors="raise")
    elif field in ["capacity_mw", "production_mwh",
                   "capacity_factor_pct", "co2_avoided_tonnes"]:
        value = float(value)
        if value < 0 or (field == "capacity_factor_pct" and value > 100):
            print("Invalid value!")
            return

    df.loc[df["record_id"] == rid, field] = value
    save_data(df)
    print("Record updated successfully!")


def delete_record():
    df = load_data()
    rid = int(input("Enter Record ID: "))

    if rid not in df["record_id"].values:
        print("Record not found!")
        return

    df = df[df["record_id"] != rid]
    save_data(df)
    print("Record deleted successfully!")


# ================= 4. NUMPY ANALYSIS =================

def numpy_analysis(df):
    arr = np.array(df["production_mwh"], dtype=float)
    arr2 = np.array(df[["capacity_mw", "production_mwh"]])

    print("\n1D Array Shape:", arr.shape)
    print("2D Array Shape:", arr2.shape)
    print("First Three Values:", arr[:3])
    print("First Row:", arr2[0])

    print("\nTotal Production:", round(np.sum(arr), 2), "MWh")
    print("Average:", round(np.mean(arr), 2), "MWh")
    print("Maximum:", round(np.max(arr), 2), "MWh")
    print("Minimum:", round(np.min(arr), 2), "MWh")
    print("Standard Deviation:", round(np.std(arr), 2))


# ================= 5. PANDAS ANALYSIS =================

def data_analysis():
    df = load_data()

    print("\nENERGY SOURCE ANALYSIS")
    source = df.groupby("energy_source").agg(
        Total_Production=("production_mwh", "sum"),
        Average_Production=("production_mwh", "mean"),
        Average_Capacity_Factor=("capacity_factor_pct", "mean"),
        Total_CO2_Avoided=("co2_avoided_tonnes", "sum")
    ).sort_values("Total_Production", ascending=False)

    print(source.round(2))

    print("\nREGIONAL ANALYSIS")
    print(df.groupby("region")["production_mwh"].sum().sort_values(ascending=False))

    print("\nMONTHLY ANALYSIS")
    df["month"] = df["date"].dt.month
    print(df.groupby("month")["production_mwh"].sum())

    print("\nSTATISTICAL SUMMARY")
    print(df[["capacity_mw", "production_mwh",
              "capacity_factor_pct", "co2_avoided_tonnes"]].describe())

    numpy_analysis(df)

    print("\nKEY INSIGHTS")
    print("Total Production:", round(df["production_mwh"].sum(), 2), "MWh")
    print("Total CO2 Avoided:", round(df["co2_avoided_tonnes"].sum(), 2), "Tonnes")
    print("Highest Producing Source:", source.index[0])
    print("Lowest Producing Source:", source.index[-1])


# ================= 6. FILTERING AND SORTING =================

def filter_records():
    df = load_data()
    source = input("Enter source (or press Enter to skip): ").strip()

    if source:
        df = df[df["energy_source"].str.lower() == source.lower()]

    df = df.sort_values("production_mwh", ascending=False)
    print(df.to_string(index=False) if not df.empty else "No matching records.")


# ================= 7. SUMMARY REPORT =================

def generate_report():
    df = load_data()

    report = [
        "RENEWABLE ENERGY PRODUCTION REPORT",
        "=" * 45,
        f"Total Records: {len(df)}",
        f"Total Production: {df['production_mwh'].sum():,.2f} MWh",
        f"Total CO2 Avoided: {df['co2_avoided_tonnes'].sum():,.2f} Tonnes",
        f"Average Capacity Factor: {df['capacity_factor_pct'].mean():.2f}%",
        "",
        "ENERGY SOURCE ANALYSIS",
        df.groupby("energy_source")["production_mwh"].sum().to_string()
    ]

    with open("reports/summary_report.txt", "w", encoding="utf-8") as file:
        file.write("\n".join(report))

    print("Summary report generated successfully!")


# ================= 8. MAIN PROGRAM =================

def main():
    create_dataset()

    # Python data structures
    energy_list = list(SOURCES)
    energy_tuple = tuple(REGIONS)
    energy_set = set(REGIONS)
    energy_dict = dict(zip(SOURCES, [120, 150, 200, 75, 60]))

    print("\nRENEWABLE ENERGY PRODUCTION ANALYSIS")
    print("List:", energy_list)
    print("Tuple:", energy_tuple)
    print("Set:", energy_set)
    print("Dictionary:", energy_dict)

    while True:
        print("\n========== MAIN MENU ==========")
        print("1. Display All Records")
        print("2. Search Record")
        print("3. Add Record")
        print("4. Update Record")
        print("5. Delete Record")
        print("6. Filter and Sort")
        print("7. Complete Data Analysis")
        print("8. Generate Summary Report")
        print("9. Exit")

        choice = input("Enter your choice: ")

        try:
            if choice == "1":
                display_records()
            elif choice == "2":
                search_record()
            elif choice == "3":
                add_record()
            elif choice == "4":
                update_record()
            elif choice == "5":
                delete_record()
            elif choice == "6":
                filter_records()
            elif choice == "7":
                data_analysis()
            elif choice == "8":
                generate_report()
            elif choice == "9":
                print("Thank you for using the project!")
                break
            else:
                print("Invalid choice!")

        except (ValueError, FileNotFoundError, KeyError) as error:
            print("Error:", error)


if __name__ == "__main__":
    main()
    print ("SACHIN YADAV")
    Print("Admission No.=2026B15410186")