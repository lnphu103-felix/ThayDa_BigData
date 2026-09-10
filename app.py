import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder, StandardScaler, Imputer
from pyspark.ml.regression import LinearRegression, RandomForestRegressor, GBTRegressor
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator

# Page Configuration
st.set_page_config(
    page_title="Amazon Delivery Prediction - Big Data App",
    page_icon="🚚",
    layout="wide"
)

st.title("🚚 Amazon Delivery Time Prediction & Big Data Analytics")
st.markdown("Ứng dụng phân tích dữ liệu lớn và dự báo thời gian giao hàng Amazon sử dụng **PySpark MLlib** & **Streamlit**.")

# 1. Initialize SparkSession
@st.cache_resource
def get_spark_session():
    return SparkSession.builder \
        .appName("Amazon_Delivery_Streamlit") \
        .config("spark.driver.memory", "2g") \
        .getOrCreate()

try:
    spark = get_spark_session()
except Exception as e:
    st.error(f"Không thể khởi tạo SparkSession (Cần môi trường Java/JDK): {e}")
    st.stop()

# 2. Sidebar - Load Dataset
st.sidebar.header("⚙️ Cấu hình Dữ liệu")
uploaded_file = st.sidebar.file_uploader("Tải lên file CSV (amazon_delivery.csv)", type=["csv"])

@st.cache_data
def load_data_pd(file_or_path):
    return pd.read_csv(file_or_path)

df_raw_pd = None
if uploaded_file is not None:
    df_raw_pd = load_data_pd(uploaded_file)
else:
    try:
        df_raw_pd = load_data_pd("amazon_delivery.csv")
        st.sidebar.success("Đã tìm thấy file amazon_delivery.csv trong thư mục gốc!")
    except FileNotFoundError:
        st.sidebar.warning("Chưa có file amazon_delivery.csv. Vui lòng tải file lên để tiếp tục.")
        st.info("👋 Hãy tải file CSV ở thanh bên trái (Sidebar) để trải nghiệm ứng dụng.")
        st.stop()

# Convert Pandas DF to Spark DF
df_spark = spark.createDataFrame(df_raw_pd)

# Layout Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Khám phá Dữ liệu (EDA)", 
    "🛠️ Tiền xử lý (Feature Engineering)", 
    "🤖 Huấn luyện Mô hình", 
    "🔮 Dự báo Thời gian Giao hàng"
])

# TAB 1: EDA
with tab1:
    st.subheader("1. Tổng quan & Trực quan hóa Dữ liệu")
    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Tổng số bản ghi (Rows)", f"{df_spark.count():,}")
    col_m2.metric("Số thuộc tính (Columns)", len(df_spark.columns))
    if "Delivery_Time" in df_raw_pd.columns:
        col_m3.metric("Thời gian giao TB (Phút)", f"{df_raw_pd['Delivery_Time'].mean():.2f}")

    st.markdown("### Xem trước bảng dữ liệu")
    st.dataframe(df_raw_pd.head(10), use_container_width=True)

    st.markdown("### Biểu đồ phân tích chuyên sâu")
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    sns.set_theme(style="whitegrid")

    if "Delivery_Time" in df_raw_pd.columns:
        sns.histplot(df_raw_pd["Delivery_Time"], kde=True, ax=axes[0, 0], color="orange")
        axes[0, 0].set_title("Phân bố Thời gian Giao hàng (Phút)")

    if "Traffic" in df_raw_pd.columns and "Delivery_Time" in df_raw_pd.columns:
        traffic_avg = df_raw_pd.groupby("Traffic")["Delivery_Time"].mean().reset_index()
        sns.barplot(data=traffic_avg, x="Traffic", y="Delivery_Time", ax=axes[0, 1], palette="Blues_d")
        axes[0, 1].set_title("Thời gian giao hàng trung bình theo Mật độ Giao thông")

    if "Weather" in df_raw_pd.columns and "Delivery_Time" in df_raw_pd.columns:
        weather_avg = df_raw_pd.groupby("Weather")["Delivery_Time"].mean().reset_index()
        sns.barplot(data=weather_avg, x="Weather", y="Delivery_Time", ax=axes[1, 0], palette="Greens_d")
        axes[1, 0].set_title("Thời gian giao hàng trung bình theo Thời tiết")

    if "Agent_Rating" in df_raw_pd.columns and "Delivery_Time" in df_raw_pd.columns:
        sns.scatterplot(data=df_raw_pd, x="Agent_Rating", y="Delivery_Time", ax=axes[1, 1], alpha=0.3, color="purple")
        axes[1, 1].set_title("Mối quan hệ giữa Rating Shipper vs Thời gian Giao")

    plt.tight_layout()
    st.pyplot(fig)

# TAB 2: Preprocessing logic in PySpark
@st.cache_data
def preprocess_data_spark(_df):
    df_proc = _df.withColumn("Store_Lat_Rad", F.radians(F.col("Store_Latitude"))) \
                 .withColumn("Store_Lon_Rad", F.radians(F.col("Store_Longitude"))) \
                 .withColumn("Drop_Lat_Rad", F.radians(F.col("Drop_Latitude"))) \
                 .withColumn("Drop_Lon_Rad", F.radians(F.col("Drop_Longitude")))

    df_proc = df_proc.withColumn("dlat", F.col("Drop_Lat_Rad") - F.col("Store_Lat_Rad")) \
                     .withColumn("dlon", F.col("Drop_Lon_Rad") - F.col("Store_Lon_Rad"))

    df_proc = df_proc.withColumn("a", F.pow(F.sin(F.col("dlat") / 2), 2) + 
                                   F.cos(F.col("Store_Lat_Rad")) * F.cos(F.col("Drop_Lat_Rad")) * F.pow(F.sin(F.col("dlon") / 2), 2))
    df_proc = df_proc.withColumn("Distance_km", 6371 * 2 * F.atan2(F.sqrt(F.col("a")), F.sqrt(1 - F.col("a"))))

    imputer = Imputer(inputCols=["Agent_Rating"], outputCols=["Agent_Rating_Imputed"]).setStrategy("mean")
    df_proc = imputer.fit(df_proc).transform(df_proc)
    df_proc = df_proc.na.fill({"Weather": "Unknown", "Traffic": "Unknown", "Vehicle": "Unknown", "Area": "Unknown", "Category": "Unknown"})

    cat_cols = ["Weather", "Traffic", "Vehicle", "Area", "Category"]
    indexers = [StringIndexer(inputCol=c, outputCol=f"{c}_Index", handleInvalid="keep") for c in cat_cols]
    encoders = [OneHotEncoder(inputCol=f"{c}_Index", outputCol=f"{c}_Vec") for c in cat_cols]

    num_cols = ["Agent_Age", "Agent_Rating_Imputed", "Distance_km"]
    vec_cols = [f"{c}_Vec" for c in cat_cols] + num_cols

    assembler = VectorAssembler(inputCols=vec_cols, outputCol="unscaled_features")
    scaler = StandardScaler(inputCol="unscaled_features", outputCol="features")

    pipeline = Pipeline(stages=indexers + encoders + [assembler, scaler])
    pipeline_model = pipeline.fit(df_proc)
    prepared_data = pipeline_model.transform(df_proc)

    return df_proc, prepared_data, pipeline_model

df_proc, prepared_data, pipeline_model = preprocess_data_spark(df_spark)

with tab2:
    st.subheader("2. Quy trình Biến đổi & Làm sạch Dữ liệu")
    st.write("1. **Tính toán Khoảng cách (Haversine Formula)**: Chuyển đổi tọa độ vĩ độ/kinh độ thành km.")
    st.write("2. **Xử lý Dữ liệu khuyết**: Điền giá trị trung bình cho Agent_Rating thiếu.")
    st.write("3. **Mã hóa Dữ liệu**: One-Hot Encoding cho Weather, Traffic, Vehicle, Area, Category.")
    st.write("4. **Chuẩn hóa Tỷ lệ**: StandardScaler tạo Feature Vector cho PySpark MLlib.")
    st.dataframe(df_proc.select("Store_Latitude", "Drop_Latitude", "Distance_km", "Agent_Rating_Imputed").limit(10).toPandas())

# TAB 3: Model Training
with tab3:
    st.subheader("3. Huấn luyện & Đánh giá Mô hình Machine Learning")
    st.write("Nhấp vào nút bên dưới để tiến hành huấn luyện các thuật toán PySpark MLlib trên tập dữ liệu.")

    if st.button("🚀 Bắt đầu Huấn luyện Mô hình", type="primary"):
        with st.spinner("Đang huấn luyện Linear Regression, Random Forest và Gradient-Boosted Trees..."):
            train_data, test_data = prepared_data.randomSplit([0.8, 0.2], seed=42)

            models = {
                "Linear Regression": LinearRegression(featuresCol="features", labelCol="Delivery_Time"),
                "Random Forest": RandomForestRegressor(featuresCol="features", labelCol="Delivery_Time", numTrees=30, seed=42),
                "Gradient-Boosted Trees": GBTRegressor(featuresCol="features", labelCol="Delivery_Time", maxIter=20, seed=42)
            }

            eval_rmse = RegressionEvaluator(labelCol="Delivery_Time", predictionCol="prediction", metricName="rmse")
            eval_r2 = RegressionEvaluator(labelCol="Delivery_Time", predictionCol="prediction", metricName="r2")
            eval_mae = RegressionEvaluator(labelCol="Delivery_Time", predictionCol="prediction", metricName="mae")

            results = []
            for name, model in models.items():
                m_fit = model.fit(train_data)
                preds = m_fit.transform(test_data)
                
                results.append({
                    "Model": name,
                    "RMSE": round(eval_rmse.evaluate(preds), 3),
                    "MAE": round(eval_mae.evaluate(preds), 3),
                    "R² Score": round(eval_r2.evaluate(preds), 3)
                })

            res_df = pd.DataFrame(results)
            st.session_state["model_results"] = res_df
            st.success("Huấn luyện hoàn tất!")

    if "model_results" in st.session_state:
        res_df = st.session_state["model_results"]
        st.subheader("Bảng so sánh Chỉ số Hiệu suất")
        st.dataframe(res_df, use_container_width=True)

        fig_m, ax_m = plt.subplots(1, 2, figsize=(12, 4))
        sns.barplot(data=res_df, x="Model", y="RMSE", ax=ax_m[0], palette="Oranges_d")
        ax_m[0].set_title("So sánh RMSE (Càng thấp càng tốt)")

        sns.barplot(data=res_df, x="Model", y="R² Score", ax=ax_m[1], palette="Purples_d")
        ax_m[1].set_title("So sánh R² Score (Càng gần 1 càng tốt)")

        plt.tight_layout()
        st.pyplot(fig_m)

# TAB 4: Real-time Prediction
with tab4:
    st.subheader("4. Dự báo Thời gian Giao hàng Đơn hàng Mới")
    st.write("Nhập thông tin chi tiết đơn hàng để hệ thống tính toán thời gian giao dự kiến:")

    col_a, col_b = st.columns(2)
    with col_a:
        agent_age = st.slider("Tuổi Shipper (Agent_Age)", 18, 60, 28)
        agent_rating = st.slider("Rating Shipper (Agent_Rating)", 1.0, 5.0, 4.6, 0.1)
        distance_val = st.number_input("Khoảng cách giao hàng (km)", 0.5, 50.0, 6.5)

    with col_b:
        weather_opt = st.selectbox("Thời tiết (Weather)", ["Sunny", "Stormy", "Sandstorms", "Windy", "Fog", "Cloudy"])
        traffic_opt = st.selectbox("Tình trạng Giao thông (Traffic)", ["Low", "Medium", "High", "Jam"])
        vehicle_opt = st.selectbox("Phương tiện (Vehicle)", ["motorcycle", "scooter", "electric_scooter", "bicycle"])
        area_opt = st.selectbox("Khu vực (Area)", ["Urban", "Metropolitan", "Semi-Urban"])
        category_opt = st.selectbox("Loại hàng hóa (Category)", ["Clothing", "Electronics", "Sports", "Cosmetics", "Toys"])

    if st.button("🔮 Tiến hành Dự báo", type="primary"):
        with st.spinner("Đang xử lý dữ liệu qua Spark Pipeline..."):
            sample_data = [(
                agent_age, agent_rating, distance_val, weather_opt, traffic_opt, vehicle_opt, area_opt, category_opt,
                0.0, 0.0, 0.0, 0.0, 30.0
            )]
            sample_cols = [
                "Agent_Age", "Agent_Rating", "Distance_km", "Weather", "Traffic", "Vehicle", "Area", "Category", 
                "Store_Latitude", "Store_Longitude", "Drop_Latitude", "Drop_Longitude", "Delivery_Time"
            ]
            
            single_spark_df = spark.createDataFrame(sample_data, sample_cols)
            single_spark_df = single_spark_df.withColumn("Agent_Rating_Imputed", F.col("Agent_Rating"))
            
            prep_single = pipeline_model.transform(single_spark_df)
            
            rf_full = RandomForestRegressor(featuresCol="features", labelCol="Delivery_Time", numTrees=30, seed=42)
            rf_model_full = rf_full.fit(prepared_data)
            
            pred_val = rf_model_full.transform(prep_single).select("prediction").collect()[0][0]

            st.markdown("---")
            st.metric(label="⏱️ Thời gian giao hàng dự báo (Dự kiến)", value=f"{pred_val:.1f} phút")
            st.balloons()
