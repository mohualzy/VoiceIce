# ui_components.py
import streamlit as st
import soundfile as sf
import io  
import utils 
from audio_recorder_streamlit import audio_recorder

def render_sidebar_inputs():
    """渲染侧边栏：上半部分 (数据输入区)"""
    with st.sidebar:  
        st.header("📂 拾遗冰窖") 
        uploaded_file = st.file_uploader("采撷一段寒语 (wav/mp3)", type=["wav", "mp3"]) 
        st.divider()
        
        st.subheader("🎙️ 现场采音")
        st.caption("(点击麦克风开始录音，再次点击结束)")
        recorded_audio_bytes = audio_recorder(
            text=" 录音", 
            recording_color="#FF0000", 
            neutral_color="#808080", 
            icon_size="2x",
            energy_threshold=(-1.0, 1.0), 
            pause_threshold=60.0          
        )
        st.divider()
        
    return uploaded_file, recorded_audio_bytes


def render_sidebar_history():
    """渲染侧边栏：下半部分 (历史记录区)"""
    selected_history = None
    delete_triggered = False  # 是否按下了融化按钮
    files_to_delete = []      # 收集需要融化的文件名
    
    with st.sidebar:
        st.subheader("🗂️ 流年冰迹")  
        vault = st.session_state.get('audio_vault', {})
        
        if vault:
            st.caption('"**点击聆听**"')
            for name in reversed(list(vault.keys())):
                if st.button(f"❄️ {name}", use_container_width=True):
                    selected_history = name
            
            st.divider()
            
            # --- 新增：多选融化区 ---
            st.caption('"**融化冰迹**"')
            files_to_delete = st.multiselect(
                label="选择文件",
                options=list(reversed(list(vault.keys()))),
                default=[],
                label_visibility="collapsed" # 隐藏多选框自带的标签，保持界面整洁
            )
            
            # 只有当用户确实在多选框里选中了文件时，才显示融化按钮
            if files_to_delete:
                if st.button("🔥 确认融化", type="primary", use_container_width=True):
                    delete_triggered = True
        else:
            st.caption("惟有风雪立空庭...") 
            
    # 将选择信号和删除信号一并传递给主程序状态机
    return selected_history, delete_triggered, files_to_delete

def render_header():
    """渲染主标题区"""
    st.title("🧊 言冰 (Voiceice)")
    st.caption("—— 话语凝冰, 烹煮听音")

def render_controls():
    """渲染中间的控制区"""
    st.container()   # container是一个强制容纳以下ui元素的矩形容器
    col_label, col_slider = st.columns([1, 3])   # 创建两个大小为一比三的列
    with col_label:
        st.markdown("### 🔥 掌灯予暖")
        st.caption("*(Kindle the Heart)*")
    with col_slider:
        # 这里的 0.5 - 2.0 直接对应 utils 里的倍速
        temperature = st.slider("调整心火的炽度...", 0.5, 2.0, 1.0, 0.1)
        
        if temperature > 1.2:
            st.caption("当前状态：**烈焰** (火力十足)")
        elif temperature < 0.8:
            st.caption("当前状态：**温火** (轻言细语)")
    
    return temperature

def render_analysis_report(temperature):
    """分析报告UI"""
    st.header("📊 情感手札")
    # 简单的模拟算法
    aggression = max(0, 80 * (temperature - 0.8))
    calmness = max(0, 100 - aggression)
    
    m1, m2, m3 = st.columns(3)
    m1.metric("锋芒 (Sharpness)", f"{aggression:.1f}%", delta_color="inverse") # 不知道咋翻译了，你们自己预览一下吧
    m2.metric("静气 (Calmness)", f"{calmness:.1f}%")
    m3.metric("流速 (Flow)", f"{temperature}x")
    
    msg = "言语过急，恐伤人心。" if temperature > 1.2 else "缓歌慢语，如春风化雨。" # python特有的三元运算形式（类似于C中的?运算符）
    st.info(f"💡 解语：{msg}")

def render_tabs_content(y_original, y_processed, sr, temperature):
    """渲染底部的三个标签页内容"""
    st.divider()
    tab1, tab2, tab3 = st.tabs(["🌊 见字如面", "🔬 闻声绘影", "📝 解语手札"])

    # --- Tab 1: 波形 ---
    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**🧊 初结之冰 (Original)**")
            # 切换为调用新的 plotly 绘制函数
            fig1 = utils.draw_waveform_plotly(y_original, sr, "Frozen Shape", "#87CEFA")
            # 使用 st.plotly_chart 渲染，并接管容器宽度
            st.plotly_chart(fig1, use_container_width=True) 
            st.audio(y_original, sample_rate=sr) 
            
        with c2:
            st.markdown(f"**💧 春水初生 (Temp: {temperature})**")
            plot_color = "#FF7F50" if temperature > 1.0 else "#40E0D0"
            fig2 = utils.draw_waveform_plotly(y_processed, sr, "Flowing Shape", plot_color)
            st.plotly_chart(fig2, use_container_width=True)
            
            virtual_file = io.BytesIO()
            sf.write(virtual_file, y_processed, sr, format='WAV')
            st.audio(virtual_file, format='audio/wav')

    # --- Tab 2: 声谱 ---
    with tab2:
        c3, c4 = st.columns(2)
        with c3:
            st.pyplot(utils.draw_spectrogram(y_original, sr, "Frozen Spectrum"))
        with c4:
            st.pyplot(utils.draw_spectrogram(y_processed, sr, "Melted Spectrum"))

    # --- Tab 3: 总结 ---
    with tab3:
        render_analysis_report(temperature)