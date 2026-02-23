# main.py
import streamlit as st
import librosa
import utils           # 导入我们的工具箱
import ui_components   # 导入我们的UI组件
import io
import datetime

# 1. 页面设置
st.set_page_config(page_title="言冰 Voiceice", page_icon="🧊", layout="wide")

# 2. 初始化 
if 'audio_vault' not in st.session_state:
    # 核心金库：{"文件名": 纯二进制数据}
    st.session_state['audio_vault'] = {}
if 'current_target' not in st.session_state:
    # 当前系统聚焦的目标文件名
    st.session_state['current_target'] = None
if 'last_record_bytes' not in st.session_state:
    st.session_state['last_record_bytes'] = None


uploaded_file, recorded_audio_bytes = ui_components.render_sidebar_inputs()

# 3. 核心路由逻辑，及时更新后台金库 
if recorded_audio_bytes is not None and recorded_audio_bytes != st.session_state['last_record_bytes']:
    st.session_state['last_record_bytes'] = recorded_audio_bytes
    time_str = datetime.datetime.now().strftime("%H:%M:%S")
    new_name = f"即兴心声_{time_str}.wav"
    st.session_state['audio_vault'][new_name] = recorded_audio_bytes
    st.session_state['current_target'] = new_name

elif uploaded_file is not None and uploaded_file.name not in st.session_state['audio_vault']:
    new_name = uploaded_file.name
    st.session_state['audio_vault'][new_name] = uploaded_file.getvalue()
    st.session_state['current_target'] = new_name

# 4. 渲染侧边栏的历史记录组件 (由于金库已在第二步更新，此时 UI 将精准同步)
selected_history = ui_components.render_sidebar_history()

# 补充场景：如果用户点击了历史记录按钮，切换游标
if selected_history is not None:
    st.session_state['current_target'] = selected_history


# 5. 渲染主界面并执行底层信号处理
ui_components.render_header()
target_name = st.session_state.get('current_target')

if target_name and target_name in st.session_state['audio_vault']:
    try:
        raw_bytes = st.session_state['audio_vault'][target_name]
        
        temp_path = "temp_processing.wav"
        with open(temp_path, "wb") as f:
            f.write(raw_bytes)
            
        y, sr = librosa.load(temp_path, sr=None)
        
        st.markdown(f"**当前聆听:** `{target_name}`")
        
        temperature = ui_components.render_controls()
        y_processed = utils.process_audio_speed_and_pitch(y, temperature, sr)
        
        ui_components.render_tabs_content(y, y_processed, sr, temperature)
        
    except Exception as e:
        st.error(f"处理音频时遇到干扰: {e}")
else:
    st.info("👈 请在左侧拾遗冰窖上传文件，或点击麦克风录制现场心声。")