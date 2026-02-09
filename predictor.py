import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, classification_report
)
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import shap
import pickle
import os
from streamlit_option_menu import option_menu

# 页面配置
st.set_page_config(
    page_title="机器学习预测系统",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# 侧边栏菜单
with st.sidebar:
    selected = option_menu(
        menu_title="主菜单",
        options=["数据上传", "数据探索", "模型训练", "模型预测", "模型解释"],
        icons=["cloud-upload", "bar-chart", "cpu", "magic", "lightbulb"],
        menu_icon="cast",
        default_index=0,
    )

# 初始化 session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'X_train' not in st.session_state:
    st.session_state.X_train = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_train' not in st.session_state:
    st.session_state.y_train = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'feature_names' not in st.session_state:
    st.session_state.feature_names = None

# ==================== 数据上传 ====================
if selected == "数据上传":
    st.title("📊 数据上传")
    st.markdown("---")
    
    upload_method = st.radio("选择数据来源", ["上传CSV文件", "使用示例数据"])
    
    if upload_method == "上传CSV文件":
        uploaded_file = st.file_uploader("选择CSV文件", type=['csv'])
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state.data = df
                st.success(f"✅ 数据上传成功！共 {df.shape[0]} 行，{df.shape[1]} 列")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("数据预览")
                    st.dataframe(df.head(10), use_container_width=True)
                
                with col2:
                    st.subheader("数据信息")
                    st.write(f"**行数:** {df.shape[0]}")
                    st.write(f"**列数:** {df.shape[1]}")
                    st.write(f"**缺失值:** {df.isnull().sum().sum()}")
                    
                    st.subheader("数据类型")
                    st.dataframe(pd.DataFrame({
                        '列名': df.columns,
                        '数据类型': df.dtypes.values,
                        '缺失值': df.isnull().sum().values
                    }), use_container_width=True)
                    
            except Exception as e:
                st.error(f"❌ 数据加载失败: {str(e)}")
    
    else:  # 使用示例数据
        from sklearn.datasets import make_classification
        
        if st.button("生成示例数据"):
            X, y = make_classification(
                n_samples=1000,
                n_features=10,
                n_informative=8,
                n_redundant=2,
                random_state=42
            )
            
            feature_names = [f'Feature_{i+1}' for i in range(X.shape[1])]
            df = pd.DataFrame(X, columns=feature_names)
            df['Target'] = y
            
            st.session_state.data = df
            st.success("✅ 示例数据生成成功！")
            st.dataframe(df.head(10), use_container_width=True)

# ==================== 数据探索 ====================
elif selected == "数据探索":
    st.title("📈 数据探索")
    st.markdown("---")
    
    if st.session_state.data is None:
        st.warning("⚠️ 请先上传数据！")
    else:
        df = st.session_state.data
        
        tab1, tab2, tab3 = st.tabs(["统计摘要", "数据可视化", "相关性分析"])
        
        with tab1:
            st.subheader("描述性统计")
            st.dataframe(df.describe(), use_container_width=True)
            
            # 数据质量报告
            st.subheader("数据质量报告")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("总样本数", df.shape[0])
            with col2:
                st.metric("特征数量", df.shape[1])
            with col3:
                st.metric("缺失值", df.isnull().sum().sum())
            with col4:
                st.metric("重复行", df.duplicated().sum())
        
        with tab2:
            st.subheader("特征分布")
            
            # 选择数值型列
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) > 0:
                selected_col = st.selectbox("选择特征", numeric_cols)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # 直方图 - Plotly
                    fig = px.histogram(
                        df, 
                        x=selected_col, 
                        nbins=30,
                        title=f'{selected_col} 分布',
                        color_discrete_sequence=['#636EFA']
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # 箱线图 - Plotly
                    fig = px.box(
                        df, 
                        y=selected_col,
                        title=f'{selected_col} 箱线图',
                        color_discrete_sequence=['#EF553B']
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # 如果有目标变量，显示分组分布
                if 'Target' in df.columns:
                    st.subheader("按目标变量分组")
                    fig = px.violin(
                        df, 
                        y=selected_col, 
                        x='Target',
                        box=True,
                        title=f'{selected_col} 按目标变量分组',
                        color='Target'
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.subheader("特征相关性矩阵")
            
            numeric_df = df.select_dtypes(include=[np.number])
            
            if numeric_df.shape[1] > 1:
                # 计算相关系数
                corr = numeric_df.corr()
                
                # 使用 Plotly 绘制热力图
                fig = px.imshow(
                    corr,
                    text_auto='.2f',
                    aspect='auto',
                    color_continuous_scale='RdBu_r',
                    title='特征相关性热力图'
                )
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)
                
                # 显示高相关性特征对
                st.subheader("高相关性特征对 (|r| > 0.7)")
                high_corr = []
                for i in range(len(corr.columns)):
                    for j in range(i+1, len(corr.columns)):
                        if abs(corr.iloc[i, j]) > 0.7:
                            high_corr.append({
                                '特征1': corr.columns[i],
                                '特征2': corr.columns[j],
                                '相关系数': round(corr.iloc[i, j], 3)
                            })
                
                if high_corr:
                    st.dataframe(pd.DataFrame(high_corr), use_container_width=True)
                else:
                    st.info("没有发现高相关性特征对")

# ==================== 模型训练 ====================
elif selected == "模型训练":
    st.title("🤖 模型训练")
    st.markdown("---")
    
    if st.session_state.data is None:
        st.warning("⚠️ 请先上传数据！")
    else:
        df = st.session_state.data
        
        # 配置区域
        st.subheader("1️⃣ 配置训练参数")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 选择目标变量
            target_col = st.selectbox("选择目标变量", df.columns.tolist())
            
            # 选择特征
            feature_cols = st.multiselect(
                "选择特征列",
                [col for col in df.columns if col != target_col],
                default=[col for col in df.columns if col != target_col]
            )
        
        with col2:
            # 模型选择
            model_type = st.selectbox(
                "选择模型",
                ["Random Forest", "XGBoost"]
            )
            
            # 测试集比例
            test_size = st.slider("测试集比例", 0.1, 0.4, 0.2, 0.05)
            
            # 是否使用SMOTE
            use_smote = st.checkbox("使用 SMOTE 处理不平衡数据", value=False)
        
        if st.button("🚀 开始训练", type="primary"):
            if len(feature_cols) == 0:
                st.error("❌ 请至少选择一个特征！")
            else:
                with st.spinner("模型训练中..."):
                    try:
                        # 准备数据
                        X = df[feature_cols]
                        y = df[target_col]
                        
                        # 分割数据
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=test_size, random_state=42, stratify=y
                        )
                        
                        # 保存到 session state
                        st.session_state.X_train = X_train
                        st.session_state.X_test = X_test
                        st.session_state.y_train = y_train
                        st.session_state.y_test = y_test
                        st.session_state.feature_names = feature_cols
                        
                        # SMOTE 处理
                        if use_smote:
                            smote = SMOTE(random_state=42)
                            X_train, y_train = smote.fit_resample(X_train, y_train)
                            st.info(f"✅ SMOTE 处理完成: {X_train.shape[0]} 样本")
                        
                        # 训练模型
                        if model_type == "Random Forest":
                            model = RandomForestClassifier(
                                n_estimators=100,
                                max_depth=10,
                                random_state=42,
                                n_jobs=-1
                            )
                        else:
                            model = XGBClassifier(
                                n_estimators=100,
                                max_depth=6,
                                learning_rate=0.1,
                                random_state=42,
                                n_jobs=-1,
                                eval_metric='logloss'
                            )
                        
                        model.fit(X_train, y_train)
                        st.session_state.model = model
                        
                        # 预测
                        y_pred = model.predict(X_test)
                        y_pred_proba = model.predict_proba(X_test)
                        
                        st.success("✅ 模型训练完成！")
                        
                        # 显示结果
                        st.subheader("2️⃣ 模型性能")
                        
                        # 指标
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            acc = accuracy_score(y_test, y_pred)
                            st.metric("准确率", f"{acc:.3f}")
                        
                        with col2:
                            prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                            st.metric("精确率", f"{prec:.3f}")
                        
                        with col3:
                            rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                            st.metric("召回率", f"{rec:.3f}")
                        
                        with col4:
                            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                            st.metric("F1分数", f"{f1:.3f}")
                        
                        # 混淆矩阵和ROC曲线
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("混淆矩阵")
                            cm = confusion_matrix(y_test, y_pred)
                            
                            # 使用 Plotly
                            fig = px.imshow(
                                cm,
                                text_auto=True,
                                labels=dict(x="预测值", y="真实值"),
                                x=[f'Class {i}' for i in range(cm.shape[1])],
                                y=[f'Class {i}' for i in range(cm.shape[0])],
                                color_continuous_scale='Blues'
                            )
                            fig.update_layout(height=400)
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            st.subheader("ROC 曲线")
                            
                            # 二分类情况
                            if len(np.unique(y)) == 2:
                                fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
                                roc_auc = auc(fpr, tpr)
                                
                                fig = go.Figure()
                                fig.add_trace(go.Scatter(
                                    x=fpr, y=tpr,
                                    mode='lines',
                                    name=f'ROC (AUC = {roc_auc:.3f})',
                                    line=dict(color='darkorange', width=2)
                                ))
                                fig.add_trace(go.Scatter(
                                    x=[0, 1], y=[0, 1],
                                    mode='lines',
                                    name='Random',
                                    line=dict(color='navy', width=2, dash='dash')
                                ))
                                fig.update_layout(
                                    xaxis_title='假正率 (FPR)',
                                    yaxis_title='真正率 (TPR)',
                                    height=400,
                                    showlegend=True
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.info("多分类问题，跳过 ROC 曲线绘制")
                        
                        # 特征重要性
                        st.subheader("3️⃣ 特征重要性")
                        
                        if hasattr(model, 'feature_importances_'):
                            importance_df = pd.DataFrame({
                                '特征': feature_cols,
                                '重要性': model.feature_importances_
                            }).sort_values('重要性', ascending=False)
                            
                            fig = px.bar(
                                importance_df,
                                x='重要性',
                                y='特征',
                                orientation='h',
                                title='特征重要性排名',
                                color='重要性',
                                color_continuous_scale='Viridis'
                            )
                            fig.update_layout(height=max(400, len(feature_cols) * 30))
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # 分类报告
                        st.subheader("4️⃣ 详细分类报告")
                        report = classification_report(y_test, y_pred, output_dict=True)
                        report_df = pd.DataFrame(report).transpose()
                        st.dataframe(report_df.style.highlight_max(axis=0), use_container_width=True)
                        
                        # 保存模型
                        st.subheader("5️⃣ 保存模型")
                        if st.button("💾 保存模型"):
                            with open('trained_model.pkl', 'wb') as f:
                                pickle.dump(model, f)
                            st.success("✅ 模型已保存为 trained_model.pkl")
                        
                    except Exception as e:
                        st.error(f"❌ 训练失败: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())

# ==================== 模型预测 ====================
elif selected == "模型预测":
    st.title("🔮 模型预测")
    st.markdown("---")
    
    if st.session_state.model is None:
        st.warning("⚠️ 请先训练模型！")
        
        # 上传已有模型
        st.subheader("或上传已训练的模型")
        uploaded_model = st.file_uploader("上传 .pkl 模型文件", type=['pkl'])
        
        if uploaded_model is not None:
            try:
                model = pickle.load(uploaded_model)
                st.session_state.model = model
                st.success("✅ 模型加载成功！")
                st.rerun()
            except Exception as e:
                st.error(f"❌ 模型加载失败: {str(e)}")
    
    else:
        model = st.session_state.model
        feature_names = st.session_state.feature_names
        
        st.subheader("输入特征值进行预测")
        
        # 创建输入表单
        input_data = {}
        
        # 动态生成输入框
        cols = st.columns(3)
        for idx, feature in enumerate(feature_names):
            with cols[idx % 3]:
                input_data[feature] = st.number_input(
                    f"{feature}",
                    value=0.0,
                    format="%.4f"
                )
        
        if st.button("🎯 开始预测", type="primary"):
            try:
                # 准备输入数据
                input_df = pd.DataFrame([input_data])
                
                # 预测
                prediction = model.predict(input_df)[0]
                prediction_proba = model.predict_proba(input_df)[0]
                
                # 显示结果
                st.success("✅ 预测完成！")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("预测类别", f"Class {prediction}")
                
                with col2:
                    st.metric("预测概率", f"{prediction_proba[prediction]:.2%}")
                
                # 概率分布
                st.subheader("各类别概率分布")
                proba_df = pd.DataFrame({
                    '类别': [f'Class {i}' for i in range(len(prediction_proba))],
                    '概率': prediction_proba
                })
                
                fig = px.bar(
                    proba_df,
                    x='类别',
                    y='概率',
                    title='预测概率分布',
                    color='概率',
                    color_continuous_scale='Bluered'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # 输入数据回顾
                st.subheader("输入数据")
                st.dataframe(input_df, use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ 预测失败: {str(e)}")
        
        # 批量预测
        st.markdown("---")
        st.subheader("批量预测")
        
        batch_file = st.file_uploader("上传CSV文件进行批量预测", type=['csv'])
        
        if batch_file is not None:
            try:
                batch_df = pd.read_csv(batch_file)
                
                # 检查特征
                missing_features = set(feature_names) - set(batch_df.columns)
                if missing_features:
                    st.error(f"❌ 缺少特征: {missing_features}")
                else:
                    X_batch = batch_df[feature_names]
                    predictions = model.predict(X_batch)
                    predictions_proba = model.predict_proba(X_batch)
                    
                    # 添加预测结果
                    result_df = batch_df.copy()
                    result_df['预测类别'] = predictions
                    result_df['预测概率'] = predictions_proba.max(axis=1)
                    
                    st.success(f"✅ 批量预测完成！共 {len(result_df)} 条记录")
                    st.dataframe(result_df, use_container_width=True)
                    
                    # 下载结果
                    csv = result_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 下载预测结果",
                        data=csv,
                        file_name='predictions.csv',
                        mime='text/csv'
                    )
                    
            except Exception as e:
                st.error(f"❌ 批量预测失败: {str(e)}")

# ==================== 模型解释 ====================
elif selected == "模型解释":
    st.title("💡 模型解释 (SHAP)")
    st.markdown("---")
    
    if st.session_state.model is None or st.session_state.X_test is None:
        st.warning("⚠️ 请先训练模型！")
    else:
        model = st.session_state.model
        X_test = st.session_state.X_test
        X_train = st.session_state.X_train
        
        st.info("🔍 SHAP (SHapley Additive exPlanations) 用于解释模型的预测结果")
        
        with st.spinner("计算 SHAP 值中..."):
            try:
                # 使用小样本以节省内存
                sample_size = min(100, len(X_test))
                X_sample = X_test.sample(n=sample_size, random_state=42)
                
                # 创建 SHAP explainer
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_sample)
                
                st.success("✅ SHAP 值计算完成！")
                
                # Tab 布局
                tab1, tab2, tab3 = st.tabs(["Summary Plot", "Force Plot", "Dependence Plot"])
                
                with tab1:
                    st.subheader("SHAP Summary Plot")
                    st.write("显示所有特征对模型预测的整体影响")
                    
                    # 处理多分类情况
                    if isinstance(shap_values, list):
                        class_idx = st.selectbox("选择类别", range(len(shap_values)))
                        shap_values_plot = shap_values[class_idx]
                    else:
                        shap_values_plot = shap_values
                    
                    # 绘制 Summary Plot
                    fig, ax = plt.subplots(figsize=(10, 6))
                    shap.summary_plot(shap_values_plot, X_sample, show=False)
                    st.pyplot(fig)
                    plt.close()
                
                with tab2:
                    st.subheader("SHAP Force Plot")
                    st.write("显示单个样本的预测解释")
                    
                    sample_idx = st.slider("选择样本索引", 0, len(X_sample)-1, 0)
                    
                    # Force plot
                    if isinstance(shap_values, list):
                        expected_value = explainer.expected_value[class_idx]
                        shap_val = shap_values[class_idx][sample_idx]
                    else:
                        expected_value = explainer.expected_value
                        shap_val = shap_values[sample_idx]
                    
                    fig, ax = plt.subplots(figsize=(12, 3))
                    shap.plots.waterfall(
                        shap.Explanation(
                            values=shap_val,
                            base_values=expected_value,
                            data=X_sample.iloc[sample_idx].values,
                            feature_names=X_sample.columns.tolist()
                        ),
                        show=False
                    )
                    st.pyplot(fig)
                    plt.close()
                
                with tab3:
                    st.subheader("SHAP Dependence Plot")
                    st.write("显示某个特征与模型输出之间的关系")
                    
                    feature = st.selectbox("选择特征", X_sample.columns.tolist())
                    
                    if isinstance(shap_values, list):
                        shap_values_dep = shap_values[class_idx]
                    else:
                        shap_values_dep = shap_values
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    shap.dependence_plot(
                        feature,
                        shap_values_dep,
                        X_sample,
                        show=False
                    )
                    st.pyplot(fig)
                    plt.close()
                
                # 特征重要性（基于SHAP）
                st.subheader("基于 SHAP 的特征重要性")
                
                if isinstance(shap_values, list):
                    shap_importance = np.abs(shap_values[class_idx]).mean(axis=0)
                else:
                    shap_importance = np.abs(shap_values).mean(axis=0)
                
                importance_df = pd.DataFrame({
                    '特征': X_sample.columns,
                    'SHAP重要性': shap_importance
                }).sort_values('SHAP重要性', ascending=False)
                
                fig = px.bar(
                    importance_df,
                    x='SHAP重要性',
                    y='特征',
                    orientation='h',
                    title='基于 SHAP 的特征重要性',
                    color='SHAP重要性',
                    color_continuous_scale='Reds'
                )
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ SHAP 计算失败: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
                
                st.info("""
                💡 提示：SHAP 计算可能需要较长时间或消耗较多内存。
                如果遇到问题，可以尝试：
                1. 减少样本量
                2. 使用更简单的模型
                3. 升级到付费版本获取更多资源
                """)

# 页脚
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        <p>机器学习预测系统 v1.0 | Powered by Streamlit</p>
    </div>
    """,
    unsafe_allow_html=True
)
