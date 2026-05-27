# ==========================================
# DASHBOARD ENSO-IOD-MODOKI - VERSI GITHUB ACTIONS
# ==========================================

import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium
from folium import plugins
import branca.colormap as cm
from datetime import datetime, timedelta
import io
import base64
import os
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("📡 Mengunduh data OISST 60 hari terakhir...")
print("="*60)

# ========== 1. URL OPeNDAP OISST NRT ==========
OISST_NRT_URL = "https://coastwatch.pfeg.noaa.gov/erddap/griddap/ncdcOisst21NrtAgg"

# Buka dataset
ds = xr.open_dataset(OISST_NRT_URL, decode_times=True)

# ========== 2. Ambil Data 60 HARI TERAKHIR ==========
end_date = datetime.now()
start_date = end_date - timedelta(days=60)

print(f"📅 Periode: {start_date.strftime('%d %b %Y')} - {end_date.strftime('%d %b %Y')}")

# Seleksi data
sst_anom_raw = ds['anom'].sel(
    time=slice(start_date, end_date),
    latitude=slice(-20, 20),
    longitude=slice(30, 300)
)

# Hapus dimensi 'zlev' jika ada
sst_anom = sst_anom_raw.squeeze()

print(f"✅ Data berhasil dimuat: {len(sst_anom.time)} hari")

# ========== 3. Fungsi Area Average ==========
def area_average(da, lon_dim='longitude', lat_dim='latitude'):
    weights = np.cos(np.deg2rad(da[lat_dim]))
    return da.weighted(weights).mean(dim=(lon_dim, lat_dim))

# ========== 4. Hitung Indeks ==========
print("\n📊 Menghitung indeks ENSO, IOD, dan ENSO Modoki (EMI)...")

# Wilayah untuk ENSO (NINO3.4)
regions_enso = {
    'nino34': {'longitude': slice(190, 240), 'latitude': slice(-5, 5)}
}

# Wilayah untuk IOD
regions_iod = {
    'iod_western': {'longitude': slice(50, 70), 'latitude': slice(-10, 10)},
    'iod_eastern': {'longitude': slice(90, 110), 'latitude': slice(-10, 0)}
}

# Wilayah untuk ENSO Modoki (EMI)
regions_modoki = {
    'box_a': {'longitude': slice(165, 220), 'latitude': slice(-10, 10)},
    'box_b': {'longitude': slice(250, 290), 'latitude': slice(-15, 5)},
    'box_c': {'longitude': slice(125, 145), 'latitude': slice(-10, 20)}
}

# Hitung indeks
regional_anom_enso = {}
for name, bounds in regions_enso.items():
    regional_anom_enso[name] = area_average(sst_anom.sel(**bounds))
    print(f"   {name}: selesai")

regional_anom_iod = {}
for name, bounds in regions_iod.items():
    regional_anom_iod[name] = area_average(sst_anom.sel(**bounds))
    print(f"   {name}: selesai")

regional_anom_modoki = {}
for name, bounds in regions_modoki.items():
    regional_anom_modoki[name] = area_average(sst_anom.sel(**bounds))
    print(f"   {name}: selesai")

nino34 = regional_anom_enso['nino34']
dmi = regional_anom_iod['iod_western'] - regional_anom_iod['iod_eastern']
emi = regional_anom_modoki['box_a'] - 0.5 * regional_anom_modoki['box_b'] - 0.5 * regional_anom_modoki['box_c']

print(f"✅ Semua indeks berhasil dihitung")

# ========== 5. Buat Grafik ==========
print("\n📈 Membuat grafik time series...")

fig, ax = plt.subplots(figsize=(12, 6))

# Plot garis
line1, = ax.plot(nino34.time, nino34.values, label='NINO3.4 (ENSO)', color='red', linewidth=2, marker='o', markersize=6)
line2, = ax.plot(dmi.time, dmi.values, label='DMI (IOD)', color='blue', linewidth=2, marker='s', markersize=6)
line3, = ax.plot(emi.time, emi.values, label='EMI (ENSO Modoki)', color='green', linewidth=2, marker='^', markersize=6)

# Area shading untuk threshold
ax.fill_between(nino34.time, 0.5, max(nino34.values.max(), dmi.values.max(), emi.values.max()) + 0.2,
                alpha=0.1, color='red', label='El Niño / IOD+ / Modoki+ Zone')
ax.fill_between(nino34.time, -0.5, min(nino34.values.min(), dmi.values.min(), emi.values.min()) - 0.2,
                alpha=0.1, color='blue', label='La Niña / IOD- / Modoki- Zone')

# Garis threshold
ax.axhline(0.5, linestyle='--', color='red', alpha=0.7, linewidth=1)
ax.axhline(-0.5, linestyle='--', color='blue', alpha=0.7, linewidth=1)
ax.axhline(0.4, linestyle=':', color='orange', alpha=0.7, linewidth=1)
ax.axhline(-0.4, linestyle=':', color='cyan', alpha=0.7, linewidth=1)
ax.axhline(0, linestyle='-', color='black', alpha=0.3, linewidth=0.5)

# Tambahkan label threshold
ax.text(nino34.time[-1], 0.52, 'ENSO +0.5', fontsize=8, color='red', ha='right')
ax.text(nino34.time[-1], -0.52, 'ENSO -0.5', fontsize=8, color='blue', ha='right')
ax.text(nino34.time[-1], 0.42, 'IOD +0.4', fontsize=8, color='orange', ha='right')
ax.text(nino34.time[-1], -0.42, 'IOD -0.4', fontsize=8, color='cyan', ha='right')

ax.set_xlabel('Waktu', fontsize=11)
ax.set_ylabel('Anomali SST (°C)', fontsize=11)
ax.set_title('Indeks ENSO (NINO3.4), IOD (DMI), dan ENSO Modoki (EMI) - 60 Hari Terakhir',
             fontsize=12, fontweight='bold')
ax.legend(loc='upper left', fontsize=9)
ax.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()

buf = io.BytesIO()
plt.savefig(buf, format='png', dpi=120, bbox_inches='tight', facecolor='white')
buf.seek(0)
plot_base64 = base64.b64encode(buf.read()).decode('utf-8')
plt.close()

# ========== 6. Buat Peta ==========
print("\n🗺️ Membuat peta anomali SST...")

anom_latest = sst_anom.isel(time=-1).squeeze()

lat_vals = anom_latest.latitude.values
lon_vals = anom_latest.longitude.values
anom_vals = anom_latest.values

lat_step = max(1, len(lat_vals) // 80)
lon_step = max(1, len(lon_vals) // 160)

lat_subset = lat_vals[::lat_step]
lon_subset = lon_vals[::lon_step]
anom_subset = anom_vals[::lat_step, ::lon_step]

m = folium.Map(
    location=[0, 150],
    zoom_start=3,
    tiles='CartoDB Voyager',
    control_scale=True
)

colormap = cm.LinearColormap(
    colors=['#2166ac', '#4393c3', '#92c5de', '#d1e5f0', '#f7f7f7',
            '#fddbc7', '#f4a582', '#d6604d', '#b2182b'],
    vmin=-3, vmax=3,
    caption='Anomali SST (°C)',
    index=[-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3]
)
colormap.add_to(m)

features = []
for i in range(len(lat_subset) - 1):
    for j in range(len(lon_subset) - 1):
        anom_val = float(anom_subset[i, j])
        if not np.isnan(anom_val):
            lat_min = float(lat_subset[i])
            lat_max = float(lat_subset[i + 1])
            lon_min = float(lon_subset[j])
            lon_max = float(lon_subset[j + 1])
            if abs(lat_max - lat_min) > 0 and abs(lon_max - lon_min) > 0:
                feature = {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[
                            [lon_min, lat_min],
                            [lon_max, lat_min],
                            [lon_max, lat_max],
                            [lon_min, lat_max],
                            [lon_min, lat_min]
                        ]]
                    },
                    "properties": {"anom": anom_val, "color": colormap(anom_val)}
                }
                features.append(feature)

if features:
    geojson_data = {"type": "FeatureCollection", "features": features}
    folium.GeoJson(
        geojson_data,
        style_function=lambda feature: {
            'fillColor': feature['properties']['color'],
            'color': 'none',
            'weight': 0,
            'fillOpacity': 0.7
        },
        tooltip=folium.GeoJsonTooltip(fields=['anom'], aliases=['Anomali SST:'], localize=True),
        popup=folium.GeoJsonPopup(fields=['anom'], aliases=['Anomali SST:'], localize=True)
    ).add_to(m)

# Wilayah NINO3.4
folium.Rectangle(bounds=[[-5, 190], [5, 240]], color='#dc3545', weight=2, fill=False, popup='NINO3.4').add_to(m)
# IOD
folium.Rectangle(bounds=[[-10, 50], [10, 70]], color='#007bff', weight=2, fill=False, popup='IOD Barat').add_to(m)
folium.Rectangle(bounds=[[-10, 90], [0, 110]], color='#28a745', weight=2, fill=False, popup='IOD Timur').add_to(m)
# ENSO Modoki boxes
folium.Rectangle(bounds=[[-10, 165], [10, 220]], color='#e83e8c', weight=2, fill=False, popup='Box A (Central)').add_to(m)
folium.Rectangle(bounds=[[-15, 250], [5, 290]], color='#fd7e14', weight=2, fill=False, popup='Box B (East)').add_to(m)
folium.Rectangle(bounds=[[-10, 125], [20, 145]], color='#20c997', weight=2, fill=False, popup='Box C (West)').add_to(m)

plugins.Fullscreen().add_to(m)
map_html = m._repr_html_()

# ========== 7. Buat Tabel ==========
print("\n📋 Membuat tabel data...")

df_table = pd.DataFrame({
    'Tanggal': [d.strftime('%Y-%m-%d') for d in pd.to_datetime(nino34.time.values)],
    'NINO3.4 (°C)': [float(x) for x in nino34.values],
    'DMI (°C)': [float(x) for x in dmi.values],
    'EMI (°C)': [float(x) for x in emi.values]
})
df_table = df_table.iloc[::-1]
df_table['NINO3.4 (°C)'] = df_table['NINO3.4 (°C)'].map(lambda x: f"{x:.2f}")
df_table['DMI (°C)'] = df_table['DMI (°C)'].map(lambda x: f"{x:.2f}")
df_table['EMI (°C)'] = df_table['EMI (°C)'].map(lambda x: f"{x:.2f}")
table_html = df_table.to_html(index=False, classes='data-table', border=0)

# ========== 8. Interpretasi ==========
last_nino = float(nino34.values[-1])
last_dmi = float(dmi.values[-1])
last_emi = float(emi.values[-1])

if last_nino > 0.5:
    enso_status, enso_color = "🔴 El Niño", "#dc3545"
elif last_nino < -0.5:
    enso_status, enso_color = "🔵 La Niña", "#0d6efd"
else:
    enso_status, enso_color = "⚪ Netral", "#6c757d"

if last_dmi > 0.4:
    iod_status, iod_color = "🟠 IOD Positif", "#fd7e14"
elif last_dmi < -0.4:
    iod_status, iod_color = "🔵 IOD Negatif", "#0d6efd"
else:
    iod_status, iod_color = "⚪ IOD Netral", "#6c757d"

if last_emi > 0.5:
    emi_status, emi_color = "🟣 El Niño Modoki", "#e83e8c"
elif last_emi < -0.5:
    emi_status, emi_color = "🔵 La Niña Modoki", "#20c997"
else:
    emi_status, emi_color = "⚪ Netral", "#6c757d"

# ========== 9. Gabungkan ke HTML ==========
print("\n💾 Menyimpan dashboard ke HTML...")

full_html = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Dashboard ENSO-IOD-Modoki</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f0f2f5; padding: 15px; }}
        .container {{ max-width: 1500px; margin: 0 auto; }}

        .header {{ background: linear-gradient(135deg, #1a237e, #0d47a1); color: white; padding: 12px 20px; border-radius: 12px; margin-bottom: 15px; text-align: center; }}
        .header h1 {{ font-size: 1.4rem; margin-bottom: 4px; }}
        .header p {{ font-size: 0.7rem; opacity: 0.85; }}

        .status-row {{ display: flex; gap: 12px; margin-bottom: 15px; flex-wrap: wrap; justify-content: center; }}
        .status-card {{
            background: white;
            border-radius: 10px;
            padding: 8px 12px;
            text-align: center;
            box-shadow: 0 1px 4px rgba(0,0,0,0.1);
            flex: 1;
            min-width: 120px;
        }}
        .status-value {{ font-size: 1.3rem; font-weight: bold; }}
        .status-label {{ font-size: 0.7rem; color: #666; margin-top: 4px; }}

        .dashboard-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-bottom: 15px; }}
        .card {{ background: white; border-radius: 12px; padding: 12px; box-shadow: 0 1px 4px rgba(0,0,0,0.1); }}
        .card h2 {{ font-size: 1rem; margin-bottom: 10px; border-left: 3px solid #0d47a1; padding-left: 10px; }}

        .map-container {{ height: 450px; border-radius: 8px; overflow: hidden; }}
        .map-container iframe {{ width: 100%; height: 100%; border: none; }}

        .plot-container img {{ max-width: 100%; height: auto; border-radius: 8px; }}

        .table-container {{ max-height: 280px; overflow-y: auto; margin-top: 10px; font-size: 0.7rem; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 5px 4px; text-align: center; border-bottom: 1px solid #eee; }}
        th {{ background: #1a237e; color: white; font-size: 0.7rem; position: sticky; top: 0; }}
        tr:hover {{ background: #f5f5f5; }}

        .footer {{ text-align: center; padding: 10px; font-size: 0.6rem; color: #666; border-top: 1px solid #ddd; margin-top: 10px; }}

        @media (max-width: 900px) {{
            .dashboard-grid {{ grid-template-columns: 1fr; }}
            .status-row {{ flex-direction: column; }}
        }}
    </style>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
</head>
<body>
<div class="container">
    <div class="header">
        <h1>🌊 Pemantauan ENSO, ENSO Modoki & IOD</h1>
        <p>{datetime.now().strftime('%d %b %Y, %H:%M')} | Sumber: NOAA OISST v2.1 | 60 hari terakhir</p>
        <div style="margin-top: 8px; font-size: 1.0rem; font-weight: bold; color: #ffc107;">📊 Departemen Geofisika dan Meteorologi, FMIPA, IPB</div>
    </div>

    <div class="status-row">
        <div class="status-card">
            <div class="status-label">🌡️ NINO3.4 (ENSO)</div>
            <div class="status-value" style="color: {enso_color};">{last_nino:+.2f}°C</div>
            <div class="status-label">{enso_status}</div>
        </div>
        <div class="status-card">
            <div class="status-label">🌊 DMI (IOD)</div>
            <div class="status-value" style="color: {iod_color};">{last_dmi:+.2f}°C</div>
            <div class="status-label">{iod_status}</div>
        </div>
        <div class="status-card">
            <div class="status-label">🟣 EMI (ENSO Modoki)</div>
            <div class="status-value" style="color: {emi_color};">{last_emi:+.2f}°C</div>
            <div class="status-label">{emi_status}</div>
        </div>
    </div>

    <div class="dashboard-grid">
        <div class="card">
            <h2>🗺️ Peta Anomali SST & Wilayah Indeks</h2>
            <div class="map-container">
                {map_html}
            </div>
            <p style="font-size: 0.6rem; color: #666; margin-top: 6px;">
                💡 <span style="color:#dc3545">■ NINO3.4</span> | <span style="color:#007bff">■ IOD Barat</span> | <span style="color:#28a745">■ IOD Timur</span> |
                <span style="color:#e83e8c">■ Box A</span> | <span style="color:#fd7e14">■ Box B</span> | <span style="color:#20c997">■ Box C</span>
            </p>
        </div>

        <div class="card">
            <h2>📈 Perkembangan Indeks (60 Hari Terakhir)</h2>
            <div class="plot-container">
                <img src="data:image/png;base64,{plot_base64}" alt="Grafik Indeks">
            </div>

            <h2 style="margin-top: 15px;">📋 Tabel Nilai Indeks</h2>
            <div class="table-container">
                {table_html}
            </div>
        </div>
    </div>

    <div class="footer">
        <p>EMI = SSTA_BoxA - 0.5·SSTA_BoxB - 0.5·SSTA_BoxC | Box A: 165°E-140°W, 10°S-10°N | Box B: 110°W-70°W, 15°S-5°N | Box C: 125°E-145°E, 10°S-20°N</p>
    </div>
</div>
</body>
</html>'''

# ========== 10. Simpan ke folder output (BUKAN DOWNLOAD) ==========
# Buat folder output jika belum ada
os.makedirs('output', exist_ok=True)

# Simpan dengan nama file berisi tanggal
filename_html = f"output/Monitoring_ENSO_IOD_{datetime.now().strftime('%Y-%m-%d')}.html"
with open(filename_html, 'w', encoding='utf-8') as f:
    f.write(full_html)

# Simpan juga sebagai index.html (biar selalu bisa diakses di URL utama)
with open("output/index.html", 'w', encoding='utf-8') as f:
    f.write(full_html)

print("\n" + "="*60)
print("✅ DASHBOARD SELESAI!")
print(f"📁 File tersimpan di folder output/")
print("="*60)
print("\n📊 Statistik ringkas:")
print(f"   - Periode data: {start_date.strftime('%d %b %Y')} - {end_date.strftime('%d %b %Y')}")
print(f"   - Jumlah hari: {len(nino34.time)}")
print(f"   - NINO3.4 terakhir: {last_nino:+.2f}°C ({enso_status})")
print(f"   - DMI terakhir: {last_dmi:+.2f}°C ({iod_status})")
print(f"   - EMI terakhir: {last_emi:+.2f}°C ({emi_status})")
print("="*60)
