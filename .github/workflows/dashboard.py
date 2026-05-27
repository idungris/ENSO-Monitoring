# ==========================================
# DASHBOARD ENSO-IOD-MODOKI - VERSI GITHUB ACTIONS
# Menggunakan requests + netCDF4 untuk menghindari bug xarray
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
import requests
import tempfile

warnings.filterwarnings('ignore')

print("="*60)
print("📡 Mengunduh data OISST 60 hari terakhir...")
print("="*60)

# ========== 1. Tentukan Periode ==========
end_date = datetime.now()
start_date = end_date - timedelta(days=60)

print(f"📅 Periode: {start_date.strftime('%d %b %Y')} - {end_date.strftime('%d %b %Y')}")

# ========== 2. Download Data via HTTP Request (PALING STABIL) ==========
# Format URL untuk download NetCDF langsung dari ERDDAP
# Parameter: time, latitude, longitude
# anom = SST anomaly

# Format tanggal untuk URL
start_str = start_date.strftime('%Y-%m-%d')
end_str = end_date.strftime('%Y-%m-%d')

# URL untuk download data anomali SST
# Region: latitude -20 to 20, longitude 30 to 300 (seluruh Pasifik tropis + Indian)
url = f"https://coastwatch.pfeg.noaa.gov/erddap/griddap/ncdcOisst21NrtAgg.nc?anom[({start_str}):1:({end_str})][(0):1:(0)][(-20):1:(20)][(30):1:(300)]"

print(f"🌐 Mengunduh dari: {url[:100]}...")

try:
    # Download dengan timeout dan retry
    headers = {
        'User-Agent': 'Mozilla/5.0 (compatible; ENSO-Dashboard/1.0)'
    }
    
    response = requests.get(url, headers=headers, timeout=120, stream=True)
    response.raise_for_status()  # Raise error jika status bukan 200
    
    # Simpan ke file temporary
    with tempfile.NamedTemporaryFile(delete=False, suffix='.nc') as tmp_file:
        for chunk in response.iter_content(chunk_size=8192):
            tmp_file.write(chunk)
        tmp_path = tmp_file.name
    
    print(f"✅ Download selesai, file size: {response.headers.get('content-length', 'unknown')} bytes")
    
    # Buka file NetCDF dengan xarray
    ds = xr.open_dataset(tmp_path, decode_times=True)
    
    # Ambil data anomali
    sst_anom_raw = ds['anom']
    
    # Hapus dimensi 'zlev' jika ada
    if 'zlev' in sst_anom_raw.dims:
        sst_anom = sst_anom_raw.isel(zlev=0)
    else:
        sst_anom = sst_anom_raw
    
    # Bersihkan file temporary
    os.unlink(tmp_path)
    
    print(f"✅ Data berhasil dimuat: {len(sst_anom.time)} hari")
    print(f"   Rentang longitude: {sst_anom.longitude.values.min():.0f}°E - {sst_anom.longitude.values.max():.0f}°E")
    print(f"   Rentang latitude: {sst_anom.latitude.values.min():.0f}° - {sst_anom.latitude.values.max():.0f}°")
    
except Exception as e:
    print(f"❌ Gagal mengunduh data: {e}")
    print("   Mencoba metode alternatif...")
    
    # ========== METODE ALTERNATIF 1: Gunakan netCDF4 langsung ==========
    try:
        import netCDF4 as nc
        
        print("   Menggunakan netCDF4 sebagai alternatif...")
        
        # Buka langsung via OPeNDAP
        opendap_url = "https://coastwatch.pfeg.noaa.gov/erddap/griddap/ncdcOisst21NrtAgg"
        
        ds_nc = nc.Dataset(opendap_url)
        
        # Baca variabel anomali dengan slicing
        # Cari indeks untuk rentang waktu yang diinginkan
        time_var = ds_nc.variables['time']
        time_units = time_var.units
        
        # Konversi waktu
        from netCDF4 import num2date
        all_times = num2date(time_var[:], time_units)
        
        # Cari indeks start dan end
        time_indices = []
        for i, t in enumerate(all_times):
            if start_date <= t <= end_date:
                time_indices.append(i)
        
        if len(time_indices) == 0:
            raise ValueError("Tidak ada data dalam rentang waktu yang diminta")
        
        time_slice = slice(time_indices[0], time_indices[-1] + 1)
        
        # Baca data dengan slicing
        anom_data = ds_nc.variables['anom'][time_slice, 0, :, :]
        lat_data = ds_nc.variables['latitude'][:]
        lon_data = ds_nc.variables['longitude'][:]
        
        # Filter latitude -20 to 20
        lat_mask = (lat_data >= -20) & (lat_data <= 20)
        lat_filtered = lat_data[lat_mask]
        
        # Filter longitude 30 to 300
        lon_mask = (lon_data >= 30) & (lon_data <= 300)
        lon_filtered = lon_data[lon_mask]
        
        anom_filtered = anom_data[:, lat_mask, :][:, :, lon_mask]
        
        # Konversi ke xarray
        sst_anom = xr.DataArray(
            anom_filtered,
            dims=['time', 'latitude', 'longitude'],
            coords={
                'time': all_times[time_indices],
                'latitude': lat_filtered,
                'longitude': lon_filtered
            },
            name='anom'
        )
        
        ds_nc.close()
        
        print(f"✅ Data berhasil dimuat via netCDF4: {len(sst_anom.time)} hari")
        
    except Exception as e2:
        print(f"❌ Metode alternatif juga gagal: {e2}")
        raise Exception("Semua metode download gagal. Periksa koneksi internet atau coba lagi nanti.")

# ========== 3. Fungsi Area Average ==========
def area_average(da, lon_dim='longitude', lat_dim='latitude'):
    """Hitung rata-rata area dengan bobot cos(latitude)"""
    weights = np.cos(np.deg2rad(da[lat_dim]))
    return da.weighted(weights).mean(dim=(lon_dim, lat_dim))

# ========== 4. Hitung Indeks ENSO, IOD, Modoki ==========
print("\n📊 Menghitung indeks ENSO, IOD, dan ENSO Modoki (EMI)...")

# Wilayah untuk ENSO (NINO3.4)
# 190°E-240°E = 170°W-120°W, 5°S-5°N
regions_enso = {
    'nino34': {'longitude': slice(190, 240), 'latitude': slice(-5, 5)}
}

# Wilayah untuk IOD
# Western: 50°E-70°E, 10°S-10°N
# Eastern: 90°E-110°E, 10°S-0°
regions_iod = {
    'iod_western': {'longitude': slice(50, 70), 'latitude': slice(-10, 10)},
    'iod_eastern': {'longitude': slice(90, 110), 'latitude': slice(-10, 0)}
}

# Wilayah untuk ENSO Modoki (EMI)
# Box A (Central): 165°E-140°W (165°E-220°E), 10°S-10°N
# Box B (East): 110°W-70°W (250°E-290°E), 15°S-5°N
# Box C (West): 125°E-145°E, 10°S-20°N
regions_modoki = {
    'box_a': {'longitude': slice(165, 220), 'latitude': slice(-10, 10)},
    'box_b': {'longitude': slice(250, 290), 'latitude': slice(-15, 5)},
    'box_c': {'longitude': slice(125, 145), 'latitude': slice(-10, 20)}
}

# Hitung indeks ENSO
regional_anom_enso = {}
for name, bounds in regions_enso.items():
    regional_anom_enso[name] = area_average(sst_anom.sel(**bounds))
    print(f"   {name}: selesai")

# Hitung indeks IOD
regional_anom_iod = {}
for name, bounds in regions_iod.items():
    regional_anom_iod[name] = area_average(sst_anom.sel(**bounds))
    print(f"   {name}: selesai")

# Hitung indeks Modoki
regional_anom_modoki = {}
for name, bounds in regions_modoki.items():
    regional_anom_modoki[name] = area_average(sst_anom.sel(**bounds))
    print(f"   {name}: selesai")

# DMI = Western - Eastern (IOD)
nino34 = regional_anom_enso['nino34']
dmi = regional_anom_iod['iod_western'] - regional_anom_iod['iod_eastern']
emi = regional_anom_modoki['box_a'] - 0.5 * regional_anom_modoki['box_b'] - 0.5 * regional_anom_modoki['box_c']

print(f"✅ Semua indeks berhasil dihitung")

# ========== 5. Buat Grafik Time Series ==========
print("\n📈 Membuat grafik time series...")

fig, ax = plt.subplots(figsize=(12, 6))

# Konversi ke pandas Series untuk kemudahan plotting
nino34_series = nino34.to_series()
dmi_series = dmi.to_series()
emi_series = emi.to_series()

# Plot garis
ax.plot(nino34_series.index, nino34_series.values, 
        label='NINO3.4 (ENSO)', color='red', linewidth=2, marker='o', markersize=4)
ax.plot(dmi_series.index, dmi_series.values, 
        label='DMI (IOD)', color='blue', linewidth=2, marker='s', markersize=4)
ax.plot(emi_series.index, emi_series.values, 
        label='EMI (ENSO Modoki)', color='green', linewidth=2, marker='^', markersize=4)

# Area shading untuk threshold ENSO/IOD
y_max = max(nino34_series.max(), dmi_series.max(), emi_series.max()) + 0.3
y_min = min(nino34_series.min(), dmi_series.min(), emi_series.min()) - 0.3

ax.fill_between(nino34_series.index, 0.5, y_max, alpha=0.1, color='red', label='El Niño / IOD+ Zone')
ax.fill_between(nino34_series.index, -0.5, y_min, alpha=0.1, color='blue', label='La Niña / IOD- Zone')

# Garis threshold
ax.axhline(0.5, linestyle='--', color='red', alpha=0.7, linewidth=1)
ax.axhline(-0.5, linestyle='--', color='blue', alpha=0.7, linewidth=1)
ax.axhline(0.4, linestyle=':', color='orange', alpha=0.7, linewidth=1)
ax.axhline(-0.4, linestyle=':', color='cyan', alpha=0.7, linewidth=1)
ax.axhline(0, linestyle='-', color='black', alpha=0.3, linewidth=0.5)

# Label threshold di ujung kanan
last_date = nino34_series.index[-1]
ax.text(last_date, 0.52, 'ENSO +0.5', fontsize=8, color='red', ha='right')
ax.text(last_date, -0.52, 'ENSO -0.5', fontsize=8, color='blue', ha='right')
ax.text(last_date, 0.42, 'IOD +0.4', fontsize=8, color='orange', ha='right')
ax.text(last_date, -0.42, 'IOD -0.4', fontsize=8, color='cyan', ha='right')

ax.set_xlabel('Waktu', fontsize=11)
ax.set_ylabel('Anomali SST (°C)', fontsize=11)
ax.set_title('Indeks ENSO (NINO3.4), IOD (DMI), dan ENSO Modoki (EMI) - 60 Hari Terakhir',
             fontsize=12, fontweight='bold')
ax.legend(loc='upper left', fontsize=9)
ax.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()

# Simpan grafik ke base64
buf = io.BytesIO()
plt.savefig(buf, format='png', dpi=120, bbox_inches='tight', facecolor='white')
buf.seek(0)
plot_base64 = base64.b64encode(buf.read()).decode('utf-8')
plt.close()

# ========== 6. Buat Peta Anomali SST (Data Terbaru) ==========
print("\n🗺️ Membuat peta anomali SST dengan wilayah indeks...")

try:
    # Ambil data terbaru (hari terakhir)
    anom_latest = sst_anom.isel(time=-1)
    
    # Filter longitude untuk peta (30-300)
    lon_filter_peta = (anom_latest.longitude >= 30) & (anom_latest.longitude <= 300)
    lat_filter_peta = (anom_latest.latitude >= -20) & (anom_latest.latitude <= 20)
    
    anom_filtered = anom_latest.where(lon_filter_peta & lat_filter_peta, drop=True)
    
    lat_vals = anom_filtered.latitude.values
    lon_vals = anom_filtered.longitude.values
    anom_vals = anom_filtered.values
    
    # Subsampling untuk performa
    lat_step = max(1, len(lat_vals) // 80)
    lon_step = max(1, len(lon_vals) // 160)
    
    lat_subset = lat_vals[::lat_step]
    lon_subset = lon_vals[::lon_step]
    anom_subset = anom_vals[::lat_step, ::lon_step]
    
    # Buat peta Folium
    m = folium.Map(
        location=[0, 150],
        zoom_start=3,
        tiles='CartoDB Voyager',
        control_scale=True
    )
    
    # Color map untuk anomali SST
    colormap = cm.LinearColormap(
        colors=['#2166ac', '#4393c3', '#92c5de', '#d1e5f0', '#f7f7f7',
                '#fddbc7', '#f4a582', '#d6604d', '#b2182b'],
        vmin=-3, vmax=3,
        caption='Anomali SST (°C)',
        index=[-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3]
    )
    colormap.add_to(m)
    
    # Buat grid polygons
    features = []
    for i in range(len(lat_subset) - 1):
        for j in range(len(lon_subset) - 1):
            anom_val = float(anom_subset[i, j])
            if not np.isnan(anom_val):
                lat_min = float(lat_subset[i])
                lat_max = float(lat_subset[i + 1])
                lon_min = float(lon_subset[j])
                lon_max = float(lon_subset[j + 1])
                
                if abs(lat_max - lat_min) > 0.01 and abs(lon_max - lon_min) > 0.01:
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
    
    # Gambar batas wilayah indeks
    # NINO3.4
    folium.Rectangle(bounds=[[-5, 190], [5, 240]], color='#dc3545', weight=2, fill=False, 
                     popup='NINO3.4 (ENSO)').add_to(m)
    # IOD Western
    folium.Rectangle(bounds=[[-10, 50], [10, 70]], color='#007bff', weight=2, fill=False, 
                     popup='IOD Barat').add_to(m)
    # IOD Eastern
    folium.Rectangle(bounds=[[-10, 90], [0, 110]], color='#28a745', weight=2, fill=False, 
                     popup='IOD Timur').add_to(m)
    # Modoki Box A
    folium.Rectangle(bounds=[[-10, 165], [10, 220]], color='#e83e8c', weight=2, fill=False, 
                     popup='Box A (Central Modoki)').add_to(m)
    # Modoki Box B
    folium.Rectangle(bounds=[[-15, 250], [5, 290]], color='#fd7e14', weight=2, fill=False, 
                     popup='Box B (East Modoki)').add_to(m)
    # Modoki Box C
    folium.Rectangle(bounds=[[-10, 125], [20, 145]], color='#20c997', weight=2, fill=False, 
                     popup='Box C (West Modoki)').add_to(m)
    
    plugins.Fullscreen().add_to(m)
    map_html = m._repr_html_()
    print("✅ Peta berhasil dibuat")
    
except Exception as e:
    print(f"⚠️ Gagal membuat peta: {e}")
    map_html = f"<div style='padding: 20px; text-align: center; background: #fff3cd; border-radius: 8px;'>⚠️ Peta anomali SST gagal dimuat: {str(e)[:100]}</div>"

# ========== 7. Buat Tabel Data ==========
print("\n📋 Membuat tabel data...")

df_table = pd.DataFrame({
    'Tanggal': [d.strftime('%Y-%m-%d') for d in nino34_series.index],
    'NINO3.4 (°C)': nino34_series.values,
    'DMI (°C)': dmi_series.values,
    'EMI (°C)': emi_series.values
})

# Urutkan dari terbaru ke terlama
df_table = df_table.iloc[::-1]

# Format angka
df_table['NINO3.4 (°C)'] = df_table['NINO3.4 (°C)'].map(lambda x: f"{x:.2f}")
df_table['DMI (°C)'] = df_table['DMI (°C)'].map(lambda x: f"{x:.2f}")
df_table['EMI (°C)'] = df_table['EMI (°C)'].map(lambda x: f"{x:.2f}")

table_html = df_table.to_html(index=False, classes='data-table', border=0)

# ========== 8. Interpretasi Status Terkini ==========
last_nino = float(nino34_series.values[-1])
last_dmi = float(dmi_series.values[-1])
last_emi = float(emi_series.values[-1])

# Status ENSO
if last_nino > 0.5:
    enso_status, enso_color = "🔴 El Niño", "#dc3545"
elif last_nino < -0.5:
    enso_status, enso_color = "🔵 La Niña", "#0d6efd"
else:
    enso_status, enso_color = "⚪ Netral", "#6c757d"

# Status IOD
if last_dmi > 0.4:
    iod_status, iod_color = "🟠 IOD Positif", "#fd7e14"
elif last_dmi < -0.4:
    iod_status, iod_color = "🔵 IOD Negatif", "#0d6efd"
else:
    iod_status, iod_color = "⚪ IOD Netral", "#6c757d"

# Status ENSO Modoki
if last_emi > 0.5:
    emi_status, emi_color = "🟣 El Niño Modoki", "#e83e8c"
elif last_emi < -0.5:
    emi_status, emi_color = "🔵 La Niña Modoki", "#20c997"
else:
    emi_status, emi_color = "⚪ Netral", "#6c757d"

# ========== 9. Generate HTML Dashboard ==========
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
        <p>{datetime.now().strftime('%d %b %Y, %H:%M')} WIB | Sumber: NOAA OISST v2.1 | 60 hari terakhir</p>
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
                <span style="color:#e83e8c">■ Box A (Central)</span> | <span style="color:#fd7e14">■ Box B (East)</span> | <span style="color:#20c997">■ Box C (West)</span>
            </p>
        </div>

        <div class="card">
            <h2>📈 Perkembangan Indeks (60 Hari Terakhir)</h2>
            <div class="plot-container">
                <img src="data:image/png;base64,{plot_base64}" alt="Grafik Indeks ENSO, IOD, dan Modoki">
            </div>

            <h2 style="margin-top: 15px;">📋 Tabel Nilai Indeks</h2>
            <div class="table-container">
                {table_html}
            </div>
        </div>
    </div>

    <div class="footer">
        <p><strong>EMI</strong> = SSTA_BoxA - 0.5·SSTA_BoxB - 0.5·SSTA_BoxC | Box A: 165°E-140°W, 10°S-10°N | Box B: 110°W-70°W, 15°S-5°N | Box C: 125°E-145°E, 10°S-20°N</p>
        <p>📅 Dashboard diperbarui setiap hari jam 06:00 WIB | Data real-time dari NOAA OISST v2.1</p>
    </div>
</div>
</body>
</html>'''

# ========== 10. Simpan File ==========
os.makedirs('output', exist_ok=True)

# Simpan dengan nama file berisi tanggal
filename_html = f"output/Monitoring_ENSO_IOD_{datetime.now().strftime('%Y-%m-%d')}.html"
with open(filename_html, 'w', encoding='utf-8') as f:
    f.write(full_html)

# Simpan juga sebagai index.html (untuk GitHub Pages)
with open("output/index.html", 'w', encoding='utf-8') as f:
    f.write(full_html)

print("\n" + "="*60)
print("✅ DASHBOARD SELESAI!")
print(f"📁 File tersimpan di folder output/")
print("="*60)
print("\n📊 Statistik ringkas:")
print(f"   - Periode data: {nino34_series.index[0].strftime('%d %b %Y')} - {nino34_series.index[-1].strftime('%d %b %Y')}")
print(f"   - Jumlah hari: {len(nino34_series)}")
print(f"   - NINO3.4 terakhir: {last_nino:+.2f}°C ({enso_status})")
print(f"   - DMI terakhir: {last_dmi:+.2f}°C ({iod_status})")
print(f"   - EMI terakhir: {last_emi:+.2f}°C ({emi_status})")
print("="*60)
