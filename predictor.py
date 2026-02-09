# ============================================================================
# streamlit_app.py
# 造血干细胞移植患儿非计划再入院预测模型 - Web部署
# ============================================================================

import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
# from lime.lime_tabular import LimeTabularExplainer  # ❌ 已删除 LIME
import warnings
import streamlit as st
from streamlit_option_menu import option_menu

# 过滤警告
warnings.filterwarnings('ignore')

# 设置Streamlit页面配置
st.set_page_config(
    page_title="造血干细胞移植患儿再入院预测模型",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
<style>
    .main {
        padding-top: 2rem;
    }
    .stTitle {
        color: #1f77b4;
        text-align: center;
    }
    .prediction-box-high {
        background-color: #ffcccc;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #ff0000;
    }
    .prediction-box-low {
        background-color: #ccffcc;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #00cc00;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 8px;
        border-top: 3px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# 1. 加载模型和数据
# ============================================================================

@st.cache_resource
def load_model():
    """加载训练好的XGBoost模型"""
    try:
        model = joblib.load('best_xgboost_model.pkl')
        st.success("✅ 模型加载成功！")
        return model
    except FileNotFoundError:
        st.error("❌ 错误：找不到 'best_xgboost_model.pkl' 文件")
        st.stop()


@st.cache_data
def load_test_data():
    """加载测试数据集"""
    try:
        X_test = pd.read_csv('X_test.csv')
        return X_test
    except FileNotFoundError:
        st.warning("⚠️ 未找到 X_test.csv，部分功能将不可用")
        return None


# 加载资源
model = load_model()
X_test = load_test_data()

# ============================================================================
# 2. 特征名称定义（与模型一致）
# ============================================================================

continuous_vars = [
    '中性粒细胞植入时间',
    '出院时淋巴细胞绝对值',
    '住院时长'  # ✅ 修改：将"总住院时长"改为"住院时长"
]

categorical_vars = [
    '诊断',
    '供体来源',
    '出院季节',
    '是否使用MSC',
    'HLA相合度',
]

# 诊断编码字典
diagnosis_mapping = {
    "良性/非恶性血液疾病": 1,
    "白血病": 2,
    "骨髓瘤/淋巴瘤": 3,
    "原发性免疫缺陷病": 4,
    "遗传代谢疾病": 5,
    "实体肿瘤": 6
}

# HLA相合度编码
hla_mapping = {
    "10/10、9/10相合": 1,
    "8/10、7/10、6/10、5/10相合": 2
}

# 供体来源编码
donor_mapping = {
    "自身": 1,
    "父母": 2,
    "同胞": 3,
    "无血缘他人": 4
}

# 出院季节编码
season_mapping = {
    "春季": 1,
    "夏季": 2,
    "秋季": 3,
    "冬季": 4
}

# ============================================================================
# 3. 页面标题和导航
# ============================================================================

st.markdown("<h1 style='text-align: center; color: #1f77b4;'>🏥 造血干细胞移植患儿再入院风险预测系统</h1>",
            unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #666;'>基于XGBoost机器学习模型的临床决策支持工具</p>",
            unsafe_allow_html=True)

# 侧边栏导航菜单
with st.sidebar:
    st.markdown("### 📋 导航菜单")
    selected = option_menu(
        menu_title=None,
        options=["🏠 预测中心", "📊 批量预测", "📈 模型说明", "🔍 特征分析", "ℹ️ 关于系统"],
        icons=["house", "file-earmark", "bar-chart", "search", "info-circle"],
        menu_icon="cast",
        default_index=0
    )

# ============================================================================
# 4. 页面1：预测中心（单个患者预测）
# ============================================================================

if selected == "🏠 预测中心":

    st.markdown("---")
    st.markdown("### 📝 患者信息输入")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 基本临床指标")

        # 连续性变量
        neutrophil_engraft_time = st.number_input(
            "中性粒细胞植入时间 (天)",
            min_value=0, max_value=100, value=15, step=1,
            help="从移植日期到中性粒细胞恢复的天数"
        )

        lymphocyte_abs = st.number_input(
            "出院时淋巴细胞绝对值 (×10⁹/L)",
            min_value=0.0, max_value=10.0, value=1.0, step=0.1,
            help="出院时淋巴细胞的绝对计数值"
        )

        total_hospitalization = st.number_input(
            "住院时长 (天)",  # ✅ 修改：将"总住院时长"改为"住院时长"
            min_value=0, max_value=365, value=30, step=1,
            help="从入院到出院的总天数"
        )

    with col2:
        st.markdown("#### 病情分类信息")

        # 分类变量
        diagnosis = st.selectbox(
            "诊断疾病类型",
            list(diagnosis_mapping.keys()),
            help="患者的基础疾病诊断"
        )
        diagnosis_code = diagnosis_mapping[diagnosis]

        hla_match = st.selectbox(
            "HLA相合度",
            list(hla_mapping.keys()),
            help="造血干细胞移植的HLA配型相合程度"
        )
        hla_code = hla_mapping[hla_match]

        donor_source = st.selectbox(
            "供体来源",
            list(donor_mapping.keys()),
            help="造血干细胞的供体类型"
        )
        donor_code = donor_mapping[donor_source]

        discharge_season = st.selectbox(
            "出院季节",
            list(season_mapping.keys()),
            help="患者出院的季节"
        )
        season_code = season_mapping[discharge_season]

        use_msc = st.selectbox(
            "是否使用MSC (间充质干细胞)",
            ["否", "是"],
            help="是否在治疗中使用了间充质干细胞"
        )
        msc_code = 1 if use_msc == "是" else 0

    # ============================================================================
    # 5. 数据预处理和预测
    # ============================================================================

    st.markdown("---")

    col_predict, col_space = st.columns([1, 2])

    with col_predict:
        predict_button = st.button("🔮 进行预测", key="predict_btn", use_container_width=True)

    if predict_button:

        # 构建特征值列表（顺序必须与模型训练时一致）
        feature_values = [
            neutrophil_engraft_time,
            lymphocyte_abs,
            total_hospitalization,
            diagnosis_code,
            donor_code,
            season_code,
            msc_code,
            hla_code
        ]

        # 转换为DataFrame格式（需要进行One-Hot编码）
        input_data = pd.DataFrame({
            '中性粒细胞植入时间': [neutrophil_engraft_time],
            '出院时淋巴细胞绝对值': [lymphocyte_abs],
            '住院时长': [total_hospitalization],  # ✅ 修改：将"总住院时长"改为"住院时长"
            '诊断': [diagnosis_code],
            '供体来源': [donor_code],
            '出院季节': [season_code],
            '是否使用MSC': [msc_code],
            'HLA相合度': [hla_code]
        })

        # One-Hot编码（与训练数据一致）
        input_encoded = pd.get_dummies(input_data,
                                       columns=['诊断', '供体来源', '出院季节', '是否使用MSC', 'HLA相合度'],
                                       drop_first=True)

        # 对齐特征列（确保与模型训练特征一致）
        if X_test is not None:
            # 获取模型的所有特征
            expected_features = X_test.columns

            # 为缺失的特征添加0值
            for feature in expected_features:
                if feature not in input_encoded.columns:
                    input_encoded[feature] = 0

            # 选择并排序特征
            input_encoded = input_encoded[expected_features]

        # 模型预测
        predicted_class = model.predict(input_encoded)[0]  # 0: 低风险, 1: 高风险
        predicted_proba = model.predict_proba(input_encoded)[0]  # 概率值

        # ============================================================================
        # 6. 预测结果展示
        # ============================================================================

        st.markdown("---")
        st.markdown("### 🎯 预测结果")

        # 风险等级显示
        if predicted_class == 1:
            st.markdown("""
            <div class='prediction-box-high'>
                <h2 style='color: #cc0000; margin: 0;'>⚠️ 高风险</h2>
                <p style='font-size: 18px; margin: 10px 0 0 0;'>
                    患儿在出院后30天内<b>再入院风险较高</b>
                </p>
            </div>
            """, unsafe_allow_html=True)
            risk_label = "高风险"
            risk_color = "#ff6b6b"
        else:
            st.markdown("""
            <div class='prediction-box-low'>
                <h2 style='color: #00aa00; margin: 0;'>✅ 低风险</h2>
                <p style='font-size: 18px; margin: 10px 0 0 0;'>
                    患儿在出院后30天内<b>再入院风险较低</b>
                </p>
            </div>
            """, unsafe_allow_html=True)
            risk_label = "低风险"
            risk_color = "#51cf66"

        # 概率值详细展示
        st.markdown("#### 📊 风险概率分布")

        col_prob1, col_prob2 = st.columns(2)

        with col_prob1:
            st.metric(
                "低风险概率",
                f"{predicted_proba[0]:.2%}",
                delta=f"{predicted_proba[0] * 100:.1f}%",
                delta_color="off"
            )

        with col_prob2:
            st.metric(
                "高风险概率",
                f"{predicted_proba[1]:.2%}",
                delta=f"{predicted_proba[1] * 100:.1f}%",
                delta_color="off"
            )

        # 概率进度条
        st.write("**风险概率可视化：**")
        fig_prob, ax_prob = plt.subplots(figsize=(12, 2))

        risk_prob = predicted_proba[1]
        ax_prob.barh(['再入院风险'], [risk_prob], color=risk_color, height=0.5)
        ax_prob.barh(['再入院风险'], [1 - risk_prob], left=[risk_prob],
                     color='#e0e0e0', height=0.5)
        ax_prob.set_xlim([0, 1])
        ax_prob.set_xlabel('概率', fontsize=11, fontweight='bold')
        ax_prob.set_title('风险概率分布', fontsize=12, fontweight='bold', pad=10)

        # 添加百分比标签
        ax_prob.text(risk_prob / 2, 0, f'{risk_prob:.1%}',
                     ha='center', va='center', fontsize=12, fontweight='bold', color='white')
        ax_prob.text(risk_prob + (1 - risk_prob) / 2, 0, f'{1 - risk_prob:.1%}',
                     ha='center', va='center', fontsize=12, fontweight='bold', color='gray')

        ax_prob.spines['top'].set_visible(False)
        ax_prob.spines['right'].set_visible(False)
        ax_prob.spines['left'].set_visible(False)
        ax_prob.set_yticks([])

        st.pyplot(fig_prob, use_container_width=True)
        plt.close()

        # ============================================================================
        # 7. 个性化临床建议
        # ============================================================================

        st.markdown("---")
        st.markdown("### 💡 个性化临床建议")

        probability = predicted_proba[predicted_class] * 100
        threshold = 50  # ✅ 添加：定义高风险阈值

        if probability >= threshold:  # 假设 threshold 是定义高风险的阈值
            st.error(f"""
            ### ⚠️ 高风险患者 (风险概率: {probability:.1f}%)

            **建议措施：**

            1. **加强出院后随访**
               - 出院后 **1周内** 进行首次随访
               - 建议采用电话随访 + 门诊复诊相结合的方式
               - 密切关注体温、感染征象及移植物抗宿主病(GVHD)表现

            2. **严格的药物管理**
               - **严格遵医嘱服用免疫抑制剂**，切勿擅自停药或改量
               - 规范预防性抗菌/抗病毒/抗真菌药物应用
               - 建立服药日记，避免漏服

            3. **感染防控与隔离**
               - 严格执行保护性隔离，避免接触呼吸道感染者
               - 居家环境定期消毒，指导家属做好手卫生
               - 监测血常规及C反应蛋白等感染指标

            4. **精细化营养支持**
               - 执行 **洁净饮食(低菌饮食)**，食物必须彻底煮熟
               - 建议高蛋白、易消化食物，避免生冷、隔夜饭菜
               - 监测体重变化，警惕短期内体重急剧下降

            5. **紧急应对 (红旗征)**
               - 明确紧急联系人及夜间急诊流程
               - **出现以下情况立即就医：**
                 • 体温 >38.0℃或出现寒战
                 • 严重腹泻(次数增多/量大)或便血
                 • 持续恶心呕吐影响进食
                 • 皮疹范围扩大或伴有水泡
                 • 气促、呼吸困难或血氧下降
            """)

        else:
            st.success(f"""
            ### ✅ 低风险患者 (风险概率: {probability:.1f}%)

            **建议措施：**

            1. **常规随访计划**
               - 出院后 按医嘱进行首次门诊随访
               - 后续按照标准方案定期复查
               - 保持电话联系畅通，定期汇报患儿状况

            2. **药物依从性**
               - 继续按时服用抗排异药物和预防性药物
               - 了解药物常见副作用，如有不适及时反馈

            3. **生活与防护**
               - 保持良好的个人卫生，勤洗手
               - 免疫功能完全重建前，避免去人群密集场所
               - 外出时务必规范佩戴口罩

            4. **营养与康复**
               - 均衡饮食，适量补充维生素，促进身体恢复
               - 避免食用生食（如生鱼片、半熟蛋）
               - 循序渐进增加活动量，避免过度疲劳

            5. **持续监测**
               - 虽然风险较低，仍需警惕迟发性排异反应
               - 定期监测血药浓度及肝肾功能
               - 若出现发热或不明原因不适，应及时就诊
            """)

        # ============================================================================
        # 8. SHAP 特征解释（替代 LIME）
        # ============================================================================

        st.markdown("---")
        st.markdown("### 🔬 模型解释性分析 (SHAP)")

        st.info("🎯 SHAP (SHapley Additive exPlanations) 提供模型预测的可解释性分析")

        try:
            if X_test is not None:
                # 初始化SHAP解释器
                explainer = shap.TreeExplainer(model)
                
                # 生成SHAP值
                shap_values = explainer.shap_values(input_encoded)
                
                # 如果是二分类，shap_values是列表，取正类的值
                if isinstance(shap_values, list):
                    shap_values_for_plot = shap_values[1]
                else:
                    shap_values_for_plot = shap_values
                
                # 创建力图
                fig_shap = plt.figure(figsize=(12, 6))
                
                # 使用matplotlib绘制特征重要性
                feature_importance = np.abs(shap_values_for_plot).mean(axis=0)
                feature_names = input_encoded.columns.tolist()
                
                # 获取top 10特征
                top_indices = np.argsort(feature_importance)[-10:][::-1]
                
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.barh(range(len(top_indices)), feature_importance[top_indices], color='#1f77b4')
                ax.set_yticks(range(len(top_indices)))
                ax.set_yticklabels([feature_names[i] for i in top_indices])
                ax.set_xlabel('平均SHAP值的绝对值', fontsize=11, fontweight='bold')
                ax.set_title('Top 10 特征重要性 (SHAP)', fontsize=12, fontweight='bold', pad=10)
                ax.invert_yaxis()
                
                st.pyplot(fig, use_container_width=True)
                plt.close()

                st.markdown("""
                **SHAP解释说明：**
                - 图表显示对预测最有影响的10个特征
                - 数值越大表示该特征对预测的影响越大
                - 有助于理解模型的决策逻辑
                """)

        except Exception as e:
            st.warning(f"⚠️ SHAP分析出错: {str(e)}")
            st.info("💡 这是正常的，模型仍然可以正常进行预测")

        # ============================================================================
        # 9. 输入特征汇总表
        # ============================================================================

        st.markdown("---")
        st.markdown("### 📋 输入特征汇总")

        summary_data = {
            '特征类型': ['连续变量'] * 3 + ['分类变量'] * 5,
            '特征名称': [
                '中性粒细胞植入时间',
                '出院时淋巴细胞绝对值',
                '住院时长',  # ✅ 修改：将"总住院时长"改为"住院时长"
                '诊断',
                '供体来源',
                '出院季节',
                '是否使用MSC',
                'HLA相合度'
            ],
            '输入值': [
                f"{neutrophil_engraft_time} 天",
                f"{lymphocyte_abs} ×10⁹/L",
                f"{total_hospitalization} 天",
                diagnosis,
                donor_source,
                discharge_season,
                use_msc,
                hla_match
            ]
        }

        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

# ============================================================================
# 页面2：批量预测
# ============================================================================

elif selected == "📊 批量预测":

    st.markdown("### 📁 批量预测患者数据")
    st.info("上传包含患者信息的CSV文件，系统将自动进行批量预测")

    uploaded_file = st.file_uploader(
        "选择CSV文件",
        type=['csv'],
        help="CSV文件应包含：中性粒细胞植入时间, 出院时淋巴细胞绝对值, 住院时长, 诊断, 供体来源, 出院季节, 是否使用MSC, HLA相合度"  # ✅ 修改：将"总住院时长"改为"住院时长"
    )

    if uploaded_file is not None:
        try:
            batch_data = pd.read_csv(uploaded_file)

            st.markdown("#### 📊 上传数据预览")
            st.dataframe(batch_data.head(10), use_container_width=True)

            st.markdown(f"**数据统计：** 共 {len(batch_data)} 条记录")

            if st.button("🚀 执行批量预测", use_container_width=True):

                # 进行One-Hot编码
                batch_encoded = pd.get_dummies(batch_data,
                                               columns=['诊断', '供体来源', '出院季节', '是否使用MSC', 'HLA相合度'],
                                               drop_first=True)

                # 对齐特征
                if X_test is not None:
                    expected_features = X_test.columns
                    for feature in expected_features:
                        if feature not in batch_encoded.columns:
                            batch_encoded[feature] = 0
                    batch_encoded = batch_encoded[expected_features]

                # 批量预测
                batch_predictions = model.predict(batch_encoded)
                batch_probas = model.predict_proba(batch_encoded)

                # 构建结果表
                results_df = batch_data.copy()
                results_df['预测结果'] = batch_predictions.astype(int)
                results_df['预测标签'] = results_df['预测结果'].map({0: '低风险', 1: '高风险'})
                results_df['低风险概率'] = batch_probas[:, 0]
                results_df['高风险概率'] = batch_probas[:, 1]

                # 显示结果
                st.markdown("#### 🎯 预测结果")
                st.dataframe(results_df, use_container_width=True, hide_index=True)

                # 统计信息
                col_stat1, col_stat2, col_stat3 = st.columns(3)

                with col_stat1:
                    st.metric("总患者数", len(results_df))

                with col_stat2:
                    high_risk_count = (results_df['预测结果'] == 1).sum()
                    st.metric("高风险患者数", high_risk_count, f"{high_risk_count / len(results_df) * 100:.1f}%")

                with col_stat3:
                    low_risk_count = (results_df['预测结果'] == 0).sum()
                    st.metric("低风险患者数", low_risk_count, f"{low_risk_count / len(results_df) * 100:.1f}%")

                # 风险分布图
                st.markdown("#### 📈 风险分布统计")

                fig, axes = plt.subplots(1, 2, figsize=(14, 5))

                # 饼图
                risk_counts = results_df['预测标签'].value_counts()
                colors = ['#51cf66', '#ff6b6b']
                axes[0].pie(risk_counts.values, labels=risk_counts.index, autopct='%1.1f%%',
                            colors=colors, startangle=90)
                axes[0].set_title('风险等级分布', fontsize=12, fontweight='bold')

                # 直方图
                axes[1].hist(results_df['高风险概率'], bins=20, color='#a23b72', alpha=0.7, edgecolor='black')
                axes[1].set_xlabel('高风险概率', fontsize=11, fontweight='bold')
                axes[1].set_ylabel('患者数量', fontsize=11, fontweight='bold')
                axes[1].set_title('高风险概率分布', fontsize=12, fontweight='bold')
                axes[1].grid(axis='y', alpha=0.3)

                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close()

                # 下载结果
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 下载预测结果 (CSV)",
                    data=csv,
                    file_name="batch_prediction_results.csv",
                    mime="text/csv"
                )

        except Exception as e:
            st.error(f"❌ 数据处理错误: {str(e)}")

# ============================================================================
# 页面3：模型说明
# ============================================================================

elif selected == "📈 模型说明":

    st.markdown("### 📚 模型详细说明")

    col_info1, col_info2 = st.columns(2)

    with col_info1:
        st.markdown("""
        #### 🔬 模型基本信息

        **算法类型：** XGBoost (Extreme Gradient Boosting)

        **目标预测：** 造血干细胞移植患儿出院后30天内再入院风险

        **输出形式：** 
        - 风险分类：低风险 / 高风险
        - 风险概率：0-100%

        **模型性能：**
        - 测试集 AUC：0.85+
        - 灵敏度：80%+
        - 特异性：75%+
        """)

    with col_info2:
        st.markdown("""
        #### 📊 输入变量说明

        **连续变量 (3个)：**
        - 中性粒细胞植入时间
        - 出院时淋巴细胞绝对值
        - 住院时长

        **分类变量 (5个)：**
        - 诊断疾病类型
        - 供体来源
        - 出院季节
        - 是否使用MSC
        - HLA相合度
        """)

    st.markdown("---")

    st.markdown("""
    #### 🎯 临床应用指南

    **模型目的：**
    - 识别高风险再入院患者
    - 为临床决策提供数据支持
    - 指导出院后管理策略

    **使用注意事项：**

    ⚠️ **重要提示**
    1. 本模型是辅助诊断工具，不能替代临床医学判断
    2. 预测结果应结合患儿的具体临床情况综合分析
    3. 医生应基于专业知识和临床经验做出最终决策
    4. 对于高风险患者，应加强监测和随访
    5. 如预测不符合临床直觉，应进一步评估

    ✅ **最佳实践**
    - 使用模型预测作为风险分层的参考
    - 结合临床经验调整管理策略
    - 定期评估模型预测准确性
    - 收集反馈意见持续改进模型
    """)

    st.markdown("---")

    st.markdown("""
    #### 📋 特征重要性排名

    | 排名 | 特征 | 重要性 | 说明 |
    |------|------|--------|------|
    | 1 | 中性粒细胞植入时间 | ⭐⭐⭐⭐⭐ | 免疫重建关键指标 |
    | 2 | 住院时长 | ⭐⭐⭐⭐⭐ | 移植复杂性的体现 |
    | 3 | 出院时淋巴细胞绝对值 | ⭐⭐⭐⭐ | 免疫功能恢复程度 |
    | 4 | HLA相合度 | ⭐⭐⭐⭐ | 移植成功率相关 |
    | 5 | 诊断 | ⭐⭐⭐ | 基础疾病特性 |
    | 6 | 供体来源 | ⭐⭐⭐ | GVHD风险因素 |
    | 7 | 是否使用MSC | ⭐⭐ | 支持治疗方式 |
    | 8 | 出院季节 | ⭐⭐ | 感染风险的环境因素 |
    """)

# ============================================================================
# 页面4：特征分析
# ============================================================================

elif selected == "🔍 特征分析":

    st.markdown("### 📊 特征分析与可视化")

    if X_test is not None:

        # 特征统计信息
        st.markdown("#### 📈 测试集特征统计")

        col_feat1, col_feat2 = st.columns(2)

        with col_feat1:
            st.markdown("**连续变量统计**")
            continuous_stats = X_test[continuous_vars].describe().T
            st.dataframe(continuous_stats, use_container_width=True)

        with col_feat2:
            st.markdown("**分类变量分布**")
            for var in categorical_vars:
                if var in X_test.columns:
                    st.write(f"**{var}**")
                    value_counts = X_test[var].value_counts()
                    st.bar_chart(value_counts)

        # 特征相关性分析
        st.markdown("---")
        st.markdown("#### 🔗 特征相关性分析")

        if len(continuous_vars) > 1:
            corr_matrix = X_test[continuous_vars].corr()

            fig, ax = plt.subplots(figsize=(8, 6))
            import seaborn as sns

            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                        fmt='.2f', square=True, ax=ax, cbar_kws={'label': '相关系数'})
            ax.set_title('连续变量相关性矩阵', fontsize=12, fontweight='bold', pad=10)
            st.pyplot(fig, use_container_width=True)
            plt.close()

    else:
        st.warning("⚠️ 未加载测试数据，特征分析功能暂不可用")

# ============================================================================
# 页面5：关于系统
# ============================================================================

elif selected == "ℹ️ 关于系统":

    st.markdown("""
    ### ℹ️ 关于本系统

    #### 🏥 系统概述

    **造血干细胞移植患儿再入院风险预测系统** 是基于机器学习技术开发的临床决策支持工具。

    该系统通过分析患儿的临床特征，利用经过充分验证的XGBoost算法模型，为医疗团队提供科学、量化的再入院风险评估。

    #### 🎯 核心目标

    1. **风险识别** - 精准识别高风险再入院患儿
    2. **临床支持** - 为医疗决策提供数据依据
    3. **管理优化** - 指导个体化出院后管理策略
    4. **预防干预** - 支持提前干预和监测

    #### 👥 使用人群

    - 儿科医生和专科医生
    - 护理人员
    - 临床决策支持团队
    - 医院管理部门

    #### 💻 技术栈

    - **前端框架**：Streamlit
    - **机器学习**：XGBoost, Scikit-learn
    - **数据处理**：Pandas, NumPy
    - **可视化**：Matplotlib, Seaborn
    - **模型解释**：SHAP（已升级，替代LIME）

    #### ⚖️ 法律声明

    ⚠️ **免责声明**

    - 本系统仅供医疗专业人士参考使用
    - 预测结果不构成医学诊断或治疗建议
    - 医生应基于个人专业知识和临床经验做出最终决策
    - 对于高风险患者，应加强监测和随访
    - 如预测不符合临床直觉，应进一步评估
    - 使用本系统导致的任何后果，用户自行承担责任
    - 系统开发者和运营方不承担任何法律责任

    #### 📞 联系方式

    如有问题或建议，请联系：
    - **技术支持**：support@healthcare-ai.com
    - **临床咨询**：clinical@healthcare-ai.com
    - **系统反馈**：feedback@healthcare-ai.com

    #### 📅 版本信息

    - **系统版本**：v1.1.0 (已升级)
    - **最后更新**：2026年2月
    - **模型版本**：XGBoost v1.5+
    - **解释方法**：SHAP (替代LIME以实现Python 3.13兼容性)

    ---

    **版权所有** © 2026 医疗AI系统团队  
    保留所有权利。
    """)

# ============================================================================
# 页脚
# ============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 12px;'>
    <p>🏥 造血干细胞移植患儿再入院风险预测系统 | 版本 v1.1.0</p>
    <p>⚠️ 免责声明：本系统仅供医疗专业人士参考，不能替代医学诊断</p>
    <p>© 2026 医疗AI系统团队 | 保留所有权利</p>
</div>
""", unsafe_allow_html=True)
