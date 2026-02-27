import pandas as pd
import requests
import time
import logging

## VuAnh: PLEASE DON'T RE-RUN THIS!
# ---------------- CONFIG ----------------
INPUT_CSV = "dataset_with_source_geo.csv"
OUTPUT_CSV = "dataset_with_geo.csv"
IP_COLUMN = "Destination IP Address" 
API_URL = "https://geoip.vuiz.net/geoip/ipv4"
SLEEP_SECONDS = 0.65
HEADERS = {"User-Agent": "Mozilla/5.0"}
# ---------------------------------------

logging.basicConfig(
    filename="destination_ip_to_geo_errors.log",
    filemode="a",
    level=logging.ERROR,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

df = pd.read_csv(INPUT_CSV)
df.columns = df.columns.str.strip()

df["country"] = None
df["continent_code"] = None
df["organization"] = None

for idx, ip in enumerate(df[IP_COLUMN]):
    try:
        r = requests.get(
            API_URL,
            params={"ip": ip, "format": "json"},
            headers=HEADERS,
            timeout=20
        )

        if r.status_code == 200:
            data = r.json()
            df.at[idx, "country"] = data.get("country")
            df.at[idx, "continent_code"] = data.get("continent_code")
            df.at[idx, "organization"] = data.get("organization")
        else:
            logging.error(
                f"Row {idx} | IP {ip} | HTTP {r.status_code} | Response: {r.text}"
            )

    except Exception as e:
        logging.exception(
            f"Row {idx} | IP {ip} | Exception occurred"
        )

    time.sleep(SLEEP_SECONDS)

    if idx > 0 and idx % 500 == 0:
        df.to_csv("checkpoint.csv", index=False)

df.to_csv(OUTPUT_CSV, index=False)
