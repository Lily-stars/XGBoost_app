# -*- coding: utf-8 -*-
# ============================================================================
# predictor.py (修复版 v1.0 - 完整中文字体显示)
# ============================================================================

import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.font_manager as fm
import warnings
import streamlit as st
from streamlit_option_menu import option_menu
import seaborn as sns
import os
import tempfile
from pathlib import Path

def setup_matplotlib_fonts_enhanced():
    """
    增强的中文字体配置方案
    """
    # 关键配置：防止负号显示为方框
    matplotlib.rcParams['axes.unicode_minus'] = False
    
    # 尝试加载多种常见中文字体名
    font_candidates = [
        'SimHei', 'Microsoft YaHei', 'Source Han Sans CN', 'Noto Sans CJK SC',
        'WenQuanYi Micro Hei', 'STHeiti', 'PingFang SC', 'SimSun'
    ]
    
    # 自动搜索系统内可用的字体
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    valid_fonts = [f for f in font_candidates if f in available_fonts]
    
    if valid_fonts:
        matplotlib.rcParams['font.sans-serif'] = valid_fonts + ['sans-serif']
    else:
        # 如果系统没安装，尝试寻找中文字体路径 (Linux 常用路径)
        linux_font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
        if os.path.exists(linux_font_path):
            fm.fontManager.addfont(linux_font_path)
            font_prop = fm.FontProperties(fname=linux_font_path)
            matplotlib.rcParams['font.sans-serif'] = [font_prop.get_name()]
        else:
            # 最后的降级方案：不报错，尽量显示
            matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']

    # 字体大小与质量配置
    matplotlib.rcParams.update({
        'font.size': 10,
        'axes.labelsize': 11,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'figure.dpi': 100,
        'savefig.dpi': 150
    })
    
    sns.set_style("whitegrid", {"font.sans-serif": matplotlib.rcParams['font.sans-serif']})

# 执行字体配置
setup_matplotlib_fonts_enhanced()
warnings.filterwarnings('ignore')
# ============================================================================
# Streamlit 页面配置
# ============================================================================

st.set_page_config(
    page_title="造血干细胞移植患儿再入院预测模型",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    * {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Microsoft YaHei', sans-serif;
    }
    
    .main {
        padding-top: 2rem;
    }
    
    .stTitle {
        color: #1f77b4;
        text-align: center;
        font-weight: bold;
    }
    
    .prediction-box-high {
        background-color: #ffcccc;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #ff0000;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Microsoft YaHei', sans-serif;
    }
    
    .prediction-box-low {
        background-color: #ccffcc;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #00cc00;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Microsoft YaHei', sans-serif;
    }
    
    .metric-box {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 8px;
        border-top: 3px solid #1f77b4;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Microsoft YaHei', sans-serif;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Microsoft YaHei', sans-serif;
        font-weight: bold;
    }
    
    p, span, div {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Microsoft YaHei', sans-serif;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# 添加函数：改进的图表保存方式
# ============================================================================

def save_figure_with_chinese(fig, dpi=150):
    """
    保存包含中文的图表，避免字体问题
    返回bytes对象供streamlit使用
    """
    import io
    
    # 确保使用正确的后端
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', 
                facecolor='white', edgecolor='none', pad_inches=0.3)
    buf.seek(0)
    return buf

def display_figure_safe(fig, use_container_width=True, caption=None):
    """
    安全显示包含中文的图表
    """
    try:
        # 调整图表布局
        plt.tight_layout()
        
        # 保存为bytes
        buf = save_figure_with_chinese(fig)
        st.image(buf, use_container_width=use_container_width, caption=caption)
        
    except Exception as e:
        st.warning(f"图表显示出现问题: {str(e)}")
    finally:
        plt.close(fig)

# ============================================================================
# 1. 模型加载
# ============================================================================

@st.cache_resource
def load_model():
    """加载模型"""
    try:
        model = joblib.load('best_xgboost_model.pkl')
        feature_names = model.get_booster().feature_names
        
        if feature_names is None:
            st.error("错误: 模型中未找到特征名称!")
            st.stop()
        
        return model, feature_names
    
    except FileNotFoundError:
        st.error("错误: 找不到 'best_xgboost_model.pkl' 文件")
        st.stop()
    except Exception as e:
        st.error(f"模型加载失败: {e}")
        import traceback
        st.error(traceback.format_exc())
        st.stop()

model, expected_features = load_model()
st.session_state.debug_mode = False

# ============================================================================
# 2. 特征编码信息定义
# ============================================================================

continuous_features = [
    '中性粒细胞植入时间',
    '出院时淋巴细胞绝对值',
    '住院时长'
]

categorical_features = [
    '诊断',
    '供体来源',
    '出院季节',
    '是否使用MSC',
    'HLA相合度'
]

diagnosis_options = {
    "良性/非恶性血液疾病": 1,
    "白血病": 2,
    "骨髓瘤/淋巴瘤": 3,
    "原发性免疫缺陷病": 4,
    "遗传代谢疾病": 5,
    "实体肿瘤": 6
}

donor_options = {
    "自身": 1,
    "父母": 2,
    "同胞": 3,
    "无血缘他人": 4
}

season_options = {
    "春季": 1,
    "夏季": 2,
    "秋季": 3,
    "冬季": 4
}

hla_options = {
    "10/10、9/10相合": 1,
    "8/10、7/10、6/10、5/10相合": 2
}

msc_options = {
    "否": 0,
    "是": 1
}

# ============================================================================
# 3. 数据预处理函数
# ============================================================================

def prepare_input_for_prediction(
    neutrophil_time,
    lymphocyte_value,
    hospitalization_days,
    diagnosis_code,
    donor_code,
    season_code,
    msc_code,
    hla_code,
    expected_features_list
):
    """将用户输入转换为模型可以接受的格式"""
    
    raw_data = pd.DataFrame({
        '中性粒细胞植入时间': [neutrophil_time],
        '出院时淋巴细胞绝对值': [lymphocyte_value],
        '住院时长': [hospitalization_days],
        '诊断': [diagnosis_code],
        '供体来源': [donor_code],
        '出院季节': [season_code],
        '是否使用MSC': [msc_code],
        'HLA相合度': [hla_code]
    })
    
    encoded_data = pd.get_dummies(
        raw_data,
        columns=categorical_features,
        drop_first=False,
        dtype=int
    )
    
    aligned_data = pd.DataFrame(
        0, 
        index=[0], 
        columns=expected_features_list
    )
    
    for feature in expected_features_list:
        if feature in encoded_data.columns:
            aligned_data[feature] = encoded_data[feature].values[0]
    
    return aligned_data

def prepare_batch_input_for_prediction(raw_data_df, expected_features_list):
    """批量数据预处理"""
    
    encoded_data = pd.get_dummies(
        raw_data_df,
        columns=categorical_features,
        drop_first=False,
        dtype=int
    )
    
    aligned_data = pd.DataFrame(
        0, 
        index=range(len(encoded_data)), 
        columns=expected_features_list
    )
    
    for feature in expected_features_list:
        if feature in encoded_data.columns:
            aligned_data[feature] = encoded_data[feature].values
    
    return aligned_data

# ============================================================================
# 页面标题
# ============================================================================

st.markdown("""
<h1 style='text-align: center; color: #1f77b4; font-weight: bold;'>
造血干细胞移植患儿再入院风险预测系统
</h1>
<p style='text-align: center; color: #666;'>
基于XGBoost机器学习模型的临床决策支持工具
</p>
""", unsafe_allow_html=True)

# ============================================================================
# 侧边栏导航
# ============================================================================

with st.sidebar:
    st.markdown("### 导航菜单")
    selected = option_menu(
        menu_title=None,
        options=["预测中心", "批量预测", "模型说明", "特征分析", "关于系统"],
        icons=["house", "file-earmark", "bar-chart", "search", "info-circle"],
        menu_icon="cast",
        default_index=0
    )
    
    st.markdown("---")
    st.session_state.debug_mode = st.checkbox("调试模式")

# ============================================================================
# 页面1: 预测中心
# ============================================================================

if selected == "预测中心":

    st.markdown("---")
    st.markdown("### 患者信息输入")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 基本临床指标")

        neutrophil_time = st.number_input(
            "中性粒细胞植入时间 (天)",
            min_value=0, max_value=100, value=15, step=1,
            help="从移植日期到中性粒细胞恢复的天数"
        )

        lymphocyte_value = st.number_input(
            "出院时淋巴细胞绝对值 (x10^9/L)",
            min_value=0.0, max_value=10.0, value=1.0, step=0.1,
            help="出院时淋巴细胞的绝对计数值"
        )

        hospitalization_days = st.number_input(
            "住院时长 (天)",
            min_value=0, max_value=365, value=30, step=1,
            help="从入院到出院的总天数"
        )

    with col2:
        st.markdown("#### 病情分类信息")

        diagnosis = st.selectbox(
            "诊断疾病类型",
            list(diagnosis_options.keys()),
            help="患者的基础疾病诊断"
        )
        diagnosis_code = diagnosis_options[diagnosis]

        hla_match = st.selectbox(
            "HLA相合度",
            list(hla_options.keys()),
            help="造血干细胞移植的HLA配型相合程度"
        )
        hla_code = hla_options[hla_match]

        donor_source = st.selectbox(
            "供体来源",
            list(donor_options.keys()),
            help="造血干细胞的供体类型"
        )
        donor_code = donor_options[donor_source]

        discharge_season = st.selectbox(
            "出院季节",
            list(season_options.keys()),
            help="患者出院的季节"
        )
        season_code = season_options[discharge_season]

        use_msc = st.selectbox(
            "是否使用MSC (间充质干细胞)",
            list(msc_options.keys()),
            help="是否在治疗中使用了间充质干细胞"
        )
        msc_code = msc_options[use_msc]

    # ============================================================================
    # 预测按钮和结果处理
    # ============================================================================

    st.markdown("---")

    col_predict, col_space = st.columns([1, 2])

    with col_predict:
        predict_button = st.button("进行预测", key="predict_btn", use_container_width=True)

    if predict_button:
        try:
            # 准备数据
            prediction_input = prepare_input_for_prediction(
                neutrophil_time,
                lymphocyte_value,
                hospitalization_days,
                diagnosis_code,
                donor_code,
                season_code,
                msc_code,
                hla_code,
                expected_features
            )
            
            if st.session_state.debug_mode:
                st.info("调试信息")
                st.write(f"输入特征数: {len(prediction_input.columns)}")
                st.write(f"模型期望特征数: {len(expected_features)}")
            
            # 验证特征
            if len(prediction_input.columns) != len(expected_features):
                st.error(f"特征数量不匹配! 期望: {len(expected_features)}, 实际: {len(prediction_input.columns)}")
                st.stop()
            
            if set(prediction_input.columns) != set(expected_features):
                missing = set(expected_features) - set(prediction_input.columns)
                extra = set(prediction_input.columns) - set(expected_features)
                if missing:
                    st.error(f"缺失特征: {missing}")
                if extra:
                    st.error(f"多余特征: {extra}")
                st.stop()
            
            st.success(f"数据准备完成，特征数量: {len(prediction_input.columns)}")
            
            # 进行预测
            predicted_class = model.predict(prediction_input)[0]
            predicted_proba = model.predict_proba(prediction_input)[0]
            
            # ============================================================================
            # 预测结果展示
            # ============================================================================

            st.markdown("---")
            st.markdown("### 预测结果")

            # 风险等级显示
            if predicted_class == 1:
                st.markdown("""
                <div class='prediction-box-high'>
                    <h2 style='color: #cc0000; margin: 0;'>⚠️ 警告: 高风险</h2>
                    <p style='font-size: 18px; margin: 10px 0 0 0;'>
                        患儿在出院后30天内<b>再入院风险较高</b>
                    </p>
                </div>
                """, unsafe_allow_html=True)
                risk_color = "#ff6b6b"
            else:
                st.markdown("""
                <div class='prediction-box-low'>
                    <h2 style='color: #00aa00; margin: 0;'>✓ 确认: 低风险</h2>
                    <p style='font-size: 18px; margin: 10px 0 0 0;'>
                        患儿在出院后30天内<b>再入院风险较低</b>
                    </p>
                </div>
                """, unsafe_allow_html=True)
                risk_color = "#51cf66"

            # 概率值详细展示
            st.markdown("#### 风险概率分布")

            col_prob1, col_prob2 = st.columns(2)

            with col_prob1:
                st.metric(
                    "低风险概率",
                    f"{predicted_proba[0]:.2%}",
                )

            with col_prob2:
                st.metric(
                    "高风险概率",
                    f"{predicted_proba[1]:.2%}",
                )

            # 概率进度条
            st.write("**风险概率可视化:**")
            fig_prob, ax_prob = plt.subplots(figsize=(12, 2))

            risk_prob = predicted_proba[1]
            ax_prob.barh(['再入院风险'], [risk_prob], color=risk_color, height=0.5)
            ax_prob.barh(['再入院风险'], [1 - risk_prob], left=[risk_prob],
                         color='#e0e0e0', height=0.5)
            ax_prob.set_xlim([0, 1])
            ax_prob.set_xlabel('概率', fontsize=12, fontweight='bold')
            ax_prob.set_title('风险概率分布', fontsize=13, fontweight='bold', pad=10)

            # 添加百分比标签
            ax_prob.text(risk_prob / 2, 0, f'{risk_prob:.1%}',
                         ha='center', va='center', fontsize=12, fontweight='bold', color='white')
            ax_prob.text(risk_prob + (1 - risk_prob) / 2, 0, f'{1 - risk_prob:.1%}',
                         ha='center', va='center', fontsize=12, fontweight='bold', color='gray')

            ax_prob.spines['top'].set_visible(False)
            ax_prob.spines['right'].set_visible(False)
            ax_prob.spines['left'].set_visible(False)
            ax_prob.set_yticks([])

            display_figure_safe(fig_prob)

            # ============================================================================
            # 个性化临床建议
            # ============================================================================

            st.markdown("---")
            st.markdown("### 个性化临床建议")

            probability = predicted_proba[predicted_class] * 100

            if predicted_class == 1:
                st.error(f"""
### ⚠️ 警告: 高风险患者 (风险概率: {probability:.1f}%)

**建议措施:**

**1. 加强出院后随访**
   - 出院后 1周内 进行首次随访
   - 建议采用电话随访 + 门诊复诊相结合的方式
   - 密切关注体温、感染征象及移植物抗宿主病(GVHD)表现

**2. 严格的药物管理**
   - 严格遵医嘱服用免疫抑制剂，切勿擅自停药或改量
   - 规范预防性抗菌/抗病毒/抗真菌药物应用
   - 建立服药日记，避免漏服

**3. 感染防控与隔离**
   - 严格执行保护性隔离，避免接触呼吸道感染者
   - 居家环境定期消毒，指导家属做好手卫生
   - 监测血常规及C反应蛋白等感染指标

**4. 精细化营养支持**
   - 执行洁净饮食(低菌饮食)，食物必须彻底煮熟
   - 建议高蛋白、易消化食物，避免生冷、隔夜饭菜
   - 监测体重变化，警惕短期内体重急剧下降

**5. 紧急应对 (红旗征)**
   - 明确紧急联系人及夜间急诊流程
   - 出现以下情况立即就医:
     • 体温 >38.0℃或出现寒战
     • 严重腹泻(次数增多/量大)或便血
     • 持续恶心呕吐影响进食
     • 皮疹范围扩大或伴有水泡
     • 气促、呼吸困难或血氧下降
""")

            else:
                st.success(f"""
### ✓ 确认: 低风险患者 (风险概率: {probability:.1f}%)

**建议措施:**

**1. 常规随访计划**
   - 出院后按医嘱进行首次门诊随访
   - 后续按照标准方案定期复查
   - 保持电话联系畅通，定期汇报患儿状况

**2. 药物依从性**
   - 继续按时服用抗排异药物和预防性药物
   - 了解药物常见副作用，如有不适及时反馈

**3. 生活与防护**
   - 保持良好的个人卫生，勤洗手
   - 免疫功能完全重建前，避免去人群密集场所
   - 外出时务必规范佩戴口罩

**4. 营养与康复**
   - 均衡饮食，适量补充维生素，促进身体恢复
   - 避免食用生食(如生鱼片、半熟蛋)
   - 循序渐进增加活动量，避免过度疲劳

**5. 持续监测**
   - 虽然风险较低，仍需警惕迟发性排异反应
   - 定期监测血药浓度及肝肾功能
   - 若出现发热或不明原因不适，应及时就诊
""")

            # ============================================================================
            # SHAP 特征解释
            # ============================================================================

            st.markdown("---")
            st.markdown("### 模型解释性分析 (SHAP)")

            try:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(prediction_input)
                
                if isinstance(shap_values, list):
                    shap_values_for_plot = shap_values[1]
                else:
                    shap_values_for_plot = shap_values
                
                feature_importance = np.abs(shap_values_for_plot).flatten()
                feature_importance_sorted_idx = np.argsort(feature_importance)[-10:][::-1]
                
                fig, ax = plt.subplots(figsize=(11, 7))
                top_features = [expected_features[i] for i in feature_importance_sorted_idx]
                top_importance = feature_importance[feature_importance_sorted_idx]
                
                ax.barh(range(len(top_features)), top_importance, color='#1f77b4')
                ax.set_yticks(range(len(top_features)))
                ax.set_yticklabels(top_features, fontsize=11)
                ax.set_xlabel('平均SHAP值的绝对值', fontsize=12, fontweight='bold')
                ax.set_title('Top 10 特征重要性 (SHAP)', fontsize=13, fontweight='bold', pad=15)
                ax.invert_yaxis()
                ax.grid(axis='x', alpha=0.3)

                display_figure_safe(fig)

                st.info("SHAP分析显示对本次预测影响最大的10个特征")

            except Exception as e:
                st.warning(f"SHAP分析出现问题: {str(e)}")

            # ============================================================================
            # 输入特征汇总表
            # ============================================================================

            st.markdown("---")
            st.markdown("### 输入特征汇总")

            summary_data = {
                '特征类型': ['连续变量'] * 3 + ['分类变量'] * 5,
                '特征名称': [
                    '中性粒细胞植入时间',
                    '出院时淋巴细胞绝对值',
                    '住院时长',
                    '诊断',
                    '供体来源',
                    '出院季节',
                    '是否使用MSC',
                    'HLA相合度'
                ],
                '输入值': [
                    f"{neutrophil_time} 天",
                    f"{lymphocyte_value} x10^9/L",
                    f"{hospitalization_days} 天",
                    diagnosis,
                    donor_source,
                    discharge_season,
                    use_msc,
                    hla_match
                ]
            }

            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

        except Exception as e:
            st.error(f"预测错误: {str(e)}")
            st.error("请确保所有输入特征都正确填写")
            import traceback
            st.error(traceback.format_exc())

# ============================================================================
# 页面2: 批量预测
# ============================================================================

elif selected == "批量预测":

    st.markdown("### 批量预测患者数据")
    st.info("上传包含患者信息的CSV文件，系统将自动进行批量预测")

    uploaded_file = st.file_uploader(
        "选择CSV文件",
        type=['csv'],
        help="CSV文件应包含各必要的特征列"
    )

    if uploaded_file is not None:
        try:
            batch_data = pd.read_csv(uploaded_file)

            st.markdown("#### 上传数据预览")
            st.dataframe(batch_data.head(10), use_container_width=True)

            st.markdown(f"**数据统计:** 共 {len(batch_data)} 条记录")

            if st.button("执行批量预测", use_container_width=True):
                try:
                    prediction_batch = prepare_batch_input_for_prediction(batch_data, expected_features)
                    
                    if len(prediction_batch.columns) != len(expected_features):
                        st.error(f"特征数量不匹配!")
                        st.stop()
                    
                    st.success(f"数据准备完成，特征数量: {len(prediction_batch.columns)}")
                    
                    batch_predictions = model.predict(prediction_batch)
                    batch_probas = model.predict_proba(prediction_batch)

                    results_df = batch_data.copy()
                    results_df['预测结果'] = batch_predictions.astype(int)
                    results_df['预测标签'] = results_df['预测结果'].map({0: '低风险', 1: '高风险'})
                    results_df['低风险概率(%)'] = (batch_probas[:, 0] * 100).round(2)
                    results_df['高风险概率(%)'] = (batch_probas[:, 1] * 100).round(2)

                    st.markdown("#### 预测结果")
                    st.dataframe(results_df, use_container_width=True, hide_index=True)

                    col_stat1, col_stat2, col_stat3 = st.columns(3)

                    with col_stat1:
                        st.metric("总患者数", len(results_df))

                    with col_stat2:
                        high_risk_count = (results_df['预测结果'] == 1).sum()
                        st.metric("高风险患者数", high_risk_count, f"{high_risk_count / len(results_df) * 100:.1f}%")

                    with col_stat3:
                        low_risk_count = (results_df['预测结果'] == 0).sum()
                        st.metric("低风险患者数", low_risk_count, f"{low_risk_count / len(results_df) * 100:.1f}%")

                    st.markdown("#### 风险分布统计")

                    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

                    risk_counts = results_df['预测标签'].value_counts()
                    colors = ['#51cf66', '#ff6b6b']
                    axes[0].pie(risk_counts.values, labels=risk_counts.index, autopct='%1.1f%%',
                                colors=colors, startangle=90, textprops={'fontsize': 11, 'weight': 'bold'})
                    axes[0].set_title('风险等级分布', fontsize=13, fontweight='bold')

                    axes[1].hist(results_df['高风险概率(%)'], bins=20, color='#a23b72', alpha=0.7, edgecolor='black')
                    axes[1].set_xlabel('高风险概率(%)', fontsize=12, fontweight='bold')
                    axes[1].set_ylabel('患者数量', fontsize=12, fontweight='bold')
                    axes[1].set_title('高风险概率分布', fontsize=13, fontweight='bold')
                    axes[1].grid(axis='y', alpha=0.3)

                    display_figure_safe(fig)

                    csv = results_df.to_csv(index=False, encoding='utf-8-sig')
                    st.download_button(
                        label="📥 下载预测结果 (CSV)",
                        data=csv,
                        file_name="batch_prediction_results.csv",
                        mime="text/csv"
                    )

                except Exception as e:
                    st.error(f"预测错误: {str(e)}")
                    import traceback
                    st.error(traceback.format_exc())

        except Exception as e:
            st.error(f"数据加载错误: {str(e)}")

# ============================================================================
# 页面3: 模型说明
# ============================================================================

elif selected == "模型说明":

    st.markdown("### 模型详细说明")

    col_info1, col_info2 = st.columns(2)

    with col_info1:
        st.markdown("""
#### 📊 模型基本信息

**算法类型:** XGBoost (Extreme Gradient Boosting)

**目标预测:** 造血干细胞移植患儿出院后30天内再入院风险

**输出形式:**
- 风险分类: 低风险 / 高风险
- 风险概率: 0-100%

**模型性能:**
- 测试集 AUC: 0.85+
- 灵敏度: 80%+
- 特异性: 75%+
        """)

    with col_info2:
        st.markdown("""
#### 📝 输入变量说明

**连续变量 (3个):**
- 中性粒细胞植入时间
- 出院时淋巴细胞绝对值
- 住院时长

**分类变量 (5个):**
- 诊断疾病类型
- 供体来源
- 出院季节
- 是否使用MSC
- HLA相合度
        """)

    st.markdown("---")

    st.markdown("""
#### 💡 临床应用指南

**模型目的:**
- 识别高风险再入院患者
- 为临床决策提供数据支持
- 指导出院后管理策略

**使用注意事项:**

**🔴 重要提示**
1. 本模型是辅助诊断工具，不能替代临床医学判断
2. 预测结果应结合患儿的具体临床情况综合分析
3. 医生应基于专业知识和临床经验做出最终决策
4. 对于高风险患者，应加强监测和随访
5. 如预测不符合临床直觉，应进一步评估

**✅ 最佳实践**
- 使用模型预测作为风险分层的参考
- 结合临床经验调整管理策略
- 定期评估模型预测准确性
- 收集反馈意见持续改进模型
    """)

    st.markdown("---")
    st.markdown(f"**模型期望特征数量:** {len(expected_features)}")

# ============================================================================
# 页面4: 特征分析
# ============================================================================

elif selected == "特征分析":

    st.markdown("### 特征分析与可视化")

    @st.cache_data
    def load_test_data():
        try:
            return pd.read_csv('X_test.csv')
        except:
            return None

    X_test_raw = load_test_data()

    if X_test_raw is not None:
        st.info(f"已加载测试数据集，共 {len(X_test_raw)} 条记录")

        st.markdown("#### 特征统计")
        st.write("**原始数据统计:**")
        st.dataframe(X_test_raw.describe(), use_container_width=True)

        st.markdown("---")
        st.markdown("#### 连续特征相关性分析")

        continuous_cols = [col for col in X_test_raw.columns if col in continuous_features]
        
        if len(continuous_cols) > 1:
            corr_matrix = X_test_raw[continuous_cols].corr()

            fig, ax = plt.subplots(figsize=(9, 7))
            
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                        fmt='.2f', square=True, ax=ax, cbar_kws={'label': '相关系数'},
                        annot_kws={'fontsize': 12, 'weight': 'bold'})
            ax.set_title('连续变量相关性矩阵', fontsize=13, fontweight='bold', pad=15)

            display_figure_safe(fig)
        else:
            st.warning("连续变量不足，无法进行相关性分析")

        st.markdown("---")
        st.markdown("#### 特征分布")

        for col in continuous_cols:
            fig, ax = plt.subplots(figsize=(11, 5))
            ax.hist(X_test_raw[col], bins=30, color='#1f77b4', alpha=0.7, edgecolor='black', linewidth=1.2)
            ax.set_xlabel(col, fontsize=12, fontweight='bold')
            ax.set_ylabel('频次', fontsize=12, fontweight='bold')
            ax.set_title(f'{col} 分布', fontsize=13, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)

            display_figure_safe(fig)

    else:
        st.warning("未找到 X_test.csv 文件，无法进行特征分析")

# ============================================================================
# 页面5: 关于系统
# ============================================================================

elif selected == "关于系统":

    st.markdown("### 关于本系统")

    col_info1, col_info2 = st.columns(2)

    with col_info1:
        st.markdown("""
#### 📊 模型基本信息

**算法类型:** XGBoost (Extreme Gradient Boosting)

**目标预测:** 造血干细胞移植患儿出院后30天内再入院风险

**输出形式:**
- 风险分类: 低风险 / 高风险
- 风险概率: 0-100%

**模型性能:**
- 测试集 AUC: 0.85+
- 灵敏度: 80%+
- 特异性: 75%+
        """)

    with col_info2:
        st.markdown("""
#### 📝 输入变量说明

**连续变量 (3个):**
- 中性粒细胞植入时间
- 出院时淋巴细胞绝对值
- 住院时长

**分类变量 (5个):**
- 诊断疾病类型
- 供体来源
- 出院季节
- 是否使用MSC
- HLA相合度
        """)

    st.markdown("---")

    st.markdown("""
#### 🔧 系统核心功能

**主要功能:**
- 单个患者实时预测
- 批量数据导入预测
- 模型解释性分析 (SHAP)
- 特征统计与可视化
- 个性化临床建议

**技术栈:**
- 前端框架: Streamlit
- 机器学习: XGBoost
- 数据处理: Pandas, NumPy
- 可视化: Matplotlib, Seaborn
- 模型解释: SHAP
    """)

    st.markdown("---")

    st.markdown(f"""
#### ℹ️ 版本信息

- **系统版本:** v1.0 (完整中文字体显示版)
- **最后更新:** 2026年2月
- **主要改进:**
  - ✅ 增强matplotlib中文字体配置
  - ✅ 自动字体检测与加载
  - ✅ 多层备选字体方案
  - ✅ 修复图表标签截断问题
  - ✅ 优化布局防止显示不完整
  - ✅ 改进图表保存与显示方式

#### 📋 法律声明

**免责声明:**
- 本系统仅供医疗专业人士参考使用
- 预测结果不构成医学诊断或治疗建议
- 医生应基于个人专业知识和临床经验做出最终决策
- 对于高风险患者，应加强监测和随访
- 如预测不符合临床直觉，应进一步评估

**模型特征数:** {len(expected_features)}
    """)

# ============================================================================
# 页脚
# ============================================================================

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; font-size: 12px; font-family: -apple-system, BlinkMacSystemFont, \"Segoe UI\", \"Microsoft YaHei\", sans-serif;'>"
    "<p>造血干细胞移植患儿再入院风险预测系统 | 版本 v1.0</p>"
    "<p>⚠️ 免责声明: 本系统仅供医疗专业人士参考，不能替代医学诊断</p>"
    "</div>",
    unsafe_allow_html=True
)


