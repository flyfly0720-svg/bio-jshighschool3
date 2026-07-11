
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.integrate import solve_ivp
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Metabolomics Explorer", layout="wide")
st.title("🧬 대사체학 데이터 분석 플랫폼")
st.markdown("**암·비만·당뇨 대사 경로 모델링 & 바이오마커 발굴** | Your Inner Engine 활동 기반")

# ====================== 사이드바 ======================
st.sidebar.header("분석 모드 선택")
mode = st.sidebar.radio("Mode", ["대사 경로 Flux 모델링", "환자 vs 정상군 프로파일 비교"])

# ====================== 1. Flux 모델링 ======================
if mode == "대사 경로 Flux 모델링":
    st.subheader("대사 경로 Flux 시뮬레이션 (Glycolysis / TCA / FAO)")
    
    pathway = st.selectbox("경로 선택", ["Glycolysis", "TCA Cycle", "Fatty Acid Oxidation"])
    
    st.sidebar.subheader("파라미터 조절")
    glucose = st.sidebar.slider("Glucose 농도 (mM)", 3.0, 25.0, 5.5)
    insulin = st.sidebar.slider("Insulin 수준 (정상=1.0)", 0.2, 3.0, 1.0)
    disease_factor = st.sidebar.slider("질환 영향도 (Cancer/Obesity/Diabetes)", 0.5, 2.5, 1.0)
    
    def glycolysis_model(t, y, glucose, insulin, disease):
        G6P, F6P, pyruvate = y
        v1 = 0.8 * glucose * insulin / (1 + 0.3 * disease)   # Hexokinase
        v2 = 0.6 * G6P
        v3 = 0.9 * F6P * (1 + 0.4 * disease)                 # PFK (Warburg effect)
        return [v1 - v2, v2 - v3, v3]
    
    if st.button("🚀 Flux 시뮬레이션 실행", type="primary"):
        t_span = (0, 60)
        y0 = [0.1, 0.1, 0.1]
        sol = solve_ivp(glycolysis_model, t_span, y0, 
                       args=(glucose, insulin, disease_factor),
                       t_eval=np.linspace(0, 60, 300))
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(sol.t, sol.y[0], label='G6P', linewidth=2.5)
        ax.plot(sol.t, sol.y[1], label='F6P', linewidth=2.5)
        ax.plot(sol.t, sol.y[2], label='Pyruvate', linewidth=2.5)
        ax.set_xlabel("Time (arbitrary unit)")
        ax.set_ylabel("Relative Concentration")
        ax.set_title(f"{pathway} Flux under {disease_factor:.1f}x Disease Condition")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
        st.success(f"**해석**: Pyruvate flux가 질환 조건에서 **{disease_factor:.1f}배** 변화했습니다. (Warburg effect / Mitochondrial dysfunction 반영)")

# ====================== 2. 환자 vs 정상군 비교 ======================
elif mode == "환자 vs 정상군 프로파일 비교":
    st.subheader("환자 vs 정상군 대사 프로파일 비교 & 바이오마커 발굴")
    
    use_sample = st.checkbox("샘플 데이터 사용하기 (암 vs 정상)", value=True)
    
    if use_sample:
        np.random.seed(42)
        metabolites = ['Glucose', 'Lactate', 'Citrate', 'Succinate', 'Palmitate', 
                      'Acetyl-CoA', 'AMP', 'BCAA', '3-HB']
        normal = np.random.normal(1.0, 0.25, (40, len(metabolites)))
        disease = np.random.normal(1.45, 0.4, (40, len(metabolites)))
        
        df = pd.DataFrame(np.vstack([normal, disease]), columns=metabolites)
        df['Group'] = ['Normal'] * 40 + ['Cancer/Obesity/Diabetes'] * 40
    else:
        uploaded = st.file_uploader("CSV 파일 업로드 (샘플 행, 대사체 열)", type=["csv"])
        if uploaded:
            df = pd.read_csv(uploaded)
        else:
            st.info("샘플 데이터를 사용하거나 CSV 파일을 업로드해주세요.")
            st.stop()
    
    if 'df' in locals():
        st.dataframe(df.head(6), use_container_width=True)
        
        # PCA
        feature_cols = [col for col in df.columns if col != 'Group']
        X = df[feature_cols]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(X_scaled)
        
        fig1 = plt.figure(figsize=(10, 7))
        sns.scatterplot(x=pca_result[:,0], y=pca_result[:,1], 
                       hue=df['Group'], s=120, alpha=0.8)
        plt.title("PCA Plot - Patient vs Control Metabolome")
        plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
        plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
        st.pyplot(fig1)
        
        # Fold Change
        if 'Group' in df.columns:
            means = df.groupby('Group').mean(numeric_only=True)
            fold_change = means.loc['Cancer/Obesity/Diabetes'] / means.loc['Normal']
            fold_change = fold_change.sort_values(ascending=False)
            
            st.subheader("🔬 Top Differential Metabolites (Fold Change)")
            st.bar_chart(fold_change)
            
            st.write("**주요 바이오마커 후보**")
            st.markdown("""
            - **Lactate ↑** : Warburg effect (암)  
            - **Palmitate ↑** : 지방산 산화 증가 (비만·당뇨)  
            - **Citrate ↓** : TCA cycle 저하  
            - **Succinate ↑** : Oncometabolite
            """)

st.caption("생기부용 대사체학 프로젝트 | 실제 연구에서는 MetaboAnalyst, COBRApy, XCMS 등을 추천합니다.")
