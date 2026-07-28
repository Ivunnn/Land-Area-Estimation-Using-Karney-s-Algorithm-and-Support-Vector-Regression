import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
import pyproj
from shapely.geometry import shape
from shapely.ops import transform
from pyproj import Geod
import joblib
import folium
from folium.plugins import Draw
from streamlit_folium import st_folium
import plotly.express as px
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

st.set_page_config(page_title="Dashboard SVR Digitasi Lahan", layout="wide")

st.title("🗺️ Dashboard Koreksi Luas Lahan (On-Screen Digitizing)")
st.markdown("Aplikasi ini menggunakan **Support Vector Regression (SVR)** dan **Algoritma Karney** untuk memprediksi dan mengoreksi luas lahan.")

# --- 1. MEMUAT MODEL ---``
@st.cache_resource
def load_models():
    svr_model = joblib.load('svr_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return svr_model, scaler

try:
    svr_model, scaler = load_models()
except Exception as e:
    st.error("⚠️ Model belum ditemukan! Pastikan file svr_model.pkl dan scaler.pkl ada.")
    st.stop()

# --- 2. PILIHAN METODE INPUT ---
st.sidebar.header("Pengaturan Input")
input_method = st.sidebar.radio(
    "Pilih Cara Input Data:",
    ("Menggambar Langsung (Digitasi)", "Upload File GeoJSON")
)

gdf = None
actual_area_col = None

if input_method == "Upload File GeoJSON":
    uploaded_file = st.sidebar.file_uploader("Upload file GeoJSON", type=["geojson"])
    if uploaded_file:
        gdf = gpd.read_file(uploaded_file)
        if gdf.crs is None or gdf.crs.to_string() != 'EPSG:4326':
            gdf = gdf.to_crs('epsg:4326')
        
        st.success(f"Berhasil memuat {len(gdf)} poligon.")
        
        # Opsi untuk memilih kolom luas BPN (untuk evaluasi error)
        kolom_tersedia = ["-- Tidak Ada --"] + list(gdf.columns)
        actual_area_col = st.sidebar.selectbox("Pilih Kolom Luas Asli (BPN) untuk Evaluasi:", kolom_tersedia)
        if actual_area_col == "-- Tidak Ada --":
            actual_area_col = None

elif input_method == "Menggambar Langsung (Digitasi)":
    st.write("### Silakan gambar poligon lahan pada peta di bawah:")
    
    # Menambahkan opsi upload GeoJSON sebagai referensi cetakan/groundtruth
    ref_file = st.file_uploader("📂 [Opsional] Upload file .geojson sebagai Panduan Overlay (Groundtruth)", type=["geojson"], key="ref_geojson")
    
    # Matikan tiles bawaan agar kita bisa menambahkan custom layer
    # Set max_zoom sangat tinggi agar bisa di-zoom maksimal
    m = folium.Map(location=[-7.332019, 112.785548], zoom_start=18, max_zoom=24, tiles=None)
    
    # Jika ada file referensi yang diupload, tambahkan ke peta
    if ref_file:
        ref_gdf = gpd.read_file(ref_file)
        if ref_gdf.crs is None or ref_gdf.crs.to_string() != 'EPSG:4326':
            ref_gdf = ref_gdf.to_crs('epsg:4326')
            
        # Membuat style garis putus-putus transparan untuk panduan
        style_panduan = lambda x: {
            'color': '#FFFF00',       # Kuning mencolok
            'weight': 2.5,            # Ketebalan garis
            'dashArray': '5, 5',      # Garis putus-putus
            'fillColor': 'transparent', # Isian transparan agar satelit tetap terlihat
            'fillOpacity': 0
        }
        
        # Ambil nama kolom untuk tooltip, tapi KECUALIKAN kolom 'geometry'
        kolom_tooltip = [col for col in ref_gdf.columns if col.lower() != 'geometry']
        
        # 1. Tambahkan poligon berserta fitur Hover (Tooltip)
        folium.GeoJson(
            ref_gdf,
            name="Panduan Groundtruth",
            style_function=style_panduan,
            # Tampilkan maksimal 3 atribut pertama (selain geometry) jika ada
            tooltip=folium.GeoJsonTooltip(fields=kolom_tooltip[:3]) if len(kolom_tooltip) > 0 else None
        ).add_to(m)
        
        # 2. Tambahkan Teks Label Permanen di atas setiap poligon
        # Mencari otomatis kolom mana yang bernama 'id'
        id_col = next((col for col in ref_gdf.columns if col.lower() == 'id'), None)
        
        if id_col:
            for _, row in ref_gdf.iterrows():
                if row.geometry is not None and not pd.isnull(row[id_col]):
                    # Gunakan representative_point() agar posisi teks PASTI berada di dalam poligon
                    pt = row.geometry.representative_point()
                    
                    # Desain teks HTML dengan CSS outline/shadow hitam tebal agar kontras
                    label_html = f'''
                        <div style="
                            font-size: 11pt; 
                            color: #FFFF00; 
                            font-weight: bold; 
                            text-shadow: 1px 1px 2px black, -1px -1px 2px black, 1px -1px 2px black, -1px 1px 2px black;
                            white-space: nowrap;
                            transform: translate(-50%, -50%);
                        ">
                            {row[id_col]}
                        </div>
                    '''
                    
                    folium.Marker(
                        location=[pt.y, pt.x],
                        icon=folium.DivIcon(html=label_html)
                    ).add_to(m)
        
        # Peta otomatis zoom ke lokasi GeoJSON referensi
        bounds = ref_gdf.total_bounds  # [minx, miny, maxx, maxy]
        m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
        st.success("Layer panduan berhasil dimuat. Silakan jiplak poligon tersebut!")

    # 1. Tambahkan Custom Basemap: Peta Dasar ATR/BPN
    folium.TileLayer(
        tiles='https://petadasar.atrbpn.go.id/main/wms/{x}/{y}/{z}',
        attr='Badan Pertanahan Nasional (ATR/BPN)',
        name='Peta Dasar ATR/BPN',
        overlay=False,
        control=True,
        max_zoom=24,
        max_native_zoom=19 # Merentangkan gambar jika zoom lebih dari 19
    ).add_to(m)

    # 2. Tambahkan Custom Basemap: Google Satellite
    folium.TileLayer(
        tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
        attr='Google',
        name='Google Satellite',
        overlay=False,
        control=True,
        max_zoom=24,
        max_native_zoom=21 # Merentangkan gambar jika zoom lebih dari 21
    ).add_to(m)
    
    # 3. Tambahkan alternatif basemap: Esri World Imagery
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri',
        name='Esri World Imagery',
        overlay=False,
        control=True,
        max_zoom=24,
        max_native_zoom=19
    ).add_to(m)

    # 4. Tambahkan OpenStreetMap biasa sebagai perbandingan
    folium.TileLayer(
        'openstreetmap', 
        name='OpenStreetMap', 
        overlay=False, 
        control=True,
        max_zoom=24
    ).add_to(m)
    
    # Aktifkan LayerControl agar user bisa ganti-ganti basemap dan melihat layer groundtruth
    folium.LayerControl(position='topright').add_to(m)
    
    # Menambahkan plugin Draw agar user bisa menggambar poligon
    draw = Draw(
        export=True,
        draw_options={'polyline': False, 'rectangle': False, 'circle': False, 'marker': False, 'circlemarker': False, 'polygon': True}
    )
    draw.add_to(m)
    
    # Tampilkan Peta: Gunakan use_container_width dan perbesar height menjadi 700px
    output = st_folium(m, use_container_width=True, height=700)
    
    if output["all_drawings"] is not None and len(output["all_drawings"]) > 0:
        # Mengubah hasil gambar (GeoJSON) menjadi GeoDataFrame
        features = output["all_drawings"]
        gdf = gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")
        st.info(f"{len(gdf)} poligon baru berhasil digambar.")


# --- 3. PEMROSESAN DAN PREDIKSI ---
if gdf is not None and len(gdf) > 0:
    if st.button("Jalankan Prediksi SVR", type="primary"):
        with st.spinner("Menghitung parameter geodesik dan morfologi..."):
            results = []
            geod = Geod(ellps='WGS84')
            transformer = pyproj.Transformer.from_crs('epsg:4326', 'epsg:3857', always_xy=True)
            
            for idx, row in gdf.iterrows():
                geom = row.geometry
                if geom.geom_type == 'MultiPolygon':
                    geom = list(geom.geoms)[0]
                if geom is None or geom.is_empty or geom.geom_type != 'Polygon':
                    continue
                    
                coords = list(geom.exterior.coords)
                if len(coords) < 3: continue
                    
                # A. Karney Area
                lons, lats = zip(*coords)
                karney_area, _ = geod.polygon_area_perimeter(lons, lats)
                karney_area = abs(karney_area)
                
                # B. Fitur Morfologi (EPSG:3857)
                geom_projected = transform(transformer.transform, geom)
                planar_area = geom_projected.area
                perimeter_planar = geom_projected.length
                vertex_count = len(coords)
                compactness = (4 * np.pi * planar_area) / (perimeter_planar ** 2) if perimeter_planar > 0 else 0
                
                min_x, min_y, max_x, max_y = geom.bounds
                w, h = max_x - min_x, max_y - min_y
                elongation = max(w, h) / min(w, h) if min(w, h) > 0 else 1
                
                actual_area = row[actual_area_col] if actual_area_col else np.nan
                
                results.append({
                    'ID': f'Poligon-{idx+1}',
                    'Luas_BPN': actual_area,
                    'karney_area': karney_area,
                    'vertex_count': vertex_count,
                    'compactness': compactness,
                    'elongation': elongation
                })
                
            df = pd.DataFrame(results)
            
        if actual_area_col:
                df['Luas_BPN'] = pd.to_numeric(df['Luas_BPN'].astype(str).str.replace(',', '.'), errors='coerce')
            
        with st.spinner("Memprediksi dengan SVR..."):
            X = df[['karney_area', 'vertex_count', 'compactness', 'elongation']]
            X_scaled = scaler.transform(X)
            df['Prediksi_SVR'] = svr_model.predict(X_scaled)
            
        st.divider()
        st.subheader("📊 Hasil Pemrosesan dan Dashboard Statistik")
        
        # --- TAB UNTUK MENAMPILKAN HASIL ---
        tab1, tab2, tab3 = st.tabs(["Tabel Data", "Visualisasi (Chart)", "Evaluasi Error"])
        
        with tab1:
            kolom_tampil = ['ID', 'karney_area', 'Prediksi_SVR']
            if actual_area_col: kolom_tampil.insert(1, 'Luas_BPN')
            st.dataframe(df[kolom_tampil].style.format("{:.2f}", subset=kolom_tampil[1:]), use_container_width=True)
            
            st.download_button(
                "Unduh Hasil (CSV)", 
                data=df.to_csv(index=False).encode('utf-8'), 
                file_name='hasil_svr.csv', mime='text/csv'
            )
            
        with tab2:
            st.write("**Perbandingan Luas (Karney vs Prediksi SVR)**")
            df_melt = df.melt(id_vars='ID', value_vars=['karney_area', 'Prediksi_SVR'], 
                              var_name='Metode', value_name='Luas (m²)')
            
            fig = px.bar(df_melt, x='ID', y='Luas (m²)', color='Metode', barmode='group',
                         color_discrete_sequence=['#FF9999', '#66B2FF'])
            st.plotly_chart(fig, use_container_width=True)
            
        with tab3:
            if actual_area_col and not df['Luas_BPN'].isnull().all():
                mape_karney = mean_absolute_percentage_error(df['Luas_BPN'], df['karney_area']) * 100
                mape_svr = mean_absolute_percentage_error(df['Luas_BPN'], df['Prediksi_SVR']) * 100
                
                col1, col2 = st.columns(2)
                col1.metric("Error (MAPE) Algoritma Karney", f"{mape_karney:.2f}%")
                col2.metric("Error (MAPE) Prediksi SVR", f"{mape_svr:.2f}%", 
                            delta=f"{mape_svr - mape_karney:.2f}% (Peningkatan Model)", delta_color="inverse")
                
                st.info("💡 Angka negatif pada warna hijau menunjukkan penurunan tingkat *error*, yang berarti SVR berhasil mengoreksi dan memperbaiki akurasi luas lahan.")
            else:
                st.warning("⚠️ Data Luas Asli (BPN) tidak tersedia atau tidak dipilih. Evaluasi error tidak dapat ditampilkan.")