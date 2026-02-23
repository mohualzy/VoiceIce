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

# 3. 渲染侧边栏，并获取上传的文件
uploaded_file, recorded_audio_bytes, selected_history = ui_components.render_sidebar()

# 4. 渲染主标题
ui_components.render_header()

if recorded_audio_bytes is not None and recorded_audio_bytes != st.session_state['last_record_bytes']:
    # 场景A：产生了新的有效录音
    st.session_state['last_record_bytes'] = recorded_audio_bytes
    time_str = datetime.datetime.now().strftime("%H:%M:%S")
    new_name = f"即兴心声_{time_str}.wav"
    # 将新录音存入金库
    st.session_state['audio_vault'][new_name] = recorded_audio_bytes
    st.session_state['current_target'] = new_name

elif uploaded_file is not None and uploaded_file.name not in st.session_state['audio_vault']:
    # 场景B：检测到新的上传文件
    new_name = uploaded_file.name
    # 使用 getvalue() 提取底层二进制流存入金库
    st.session_state['audio_vault'][new_name] = uploaded_file.getvalue()
    st.session_state['current_target'] = new_name

elif selected_history is not None:
    # 场景C：用户点击了侧边栏的历史记录按钮
    st.session_state['current_target'] = selected_history
    
# 5. 核心逻辑
target_name = st.session_state.get('current_target')

if target_name and target_name in st.session_state['audio_vault']:
    try:
        # 从金库中提取当前目标音频的纯净二进制数据
        raw_bytes = st.session_state['audio_vault'][target_name]
        
        # 将二进制流写入临时文件供 librosa 解析
        temp_path = "temp_processing.wav"
        with open(temp_path, "wb") as f:
            f.write(raw_bytes)
            
        y, sr = librosa.load(temp_path, sr=None)
        
        # UI 提示当前正在处理的文件
        st.markdown(f"**当前聆听:** `{target_name}`")
        
        temperature = ui_components.render_controls()
        y_processed = utils.process_audio_speed_and_pitch(y, temperature, sr)
        
        ui_components.render_tabs_content(y, y_processed, sr, temperature)
        
    except Exception as e:
        st.error(f"处理音频时遇到干扰: {e}")
else:
    st.info("👈 请在左侧拾遗冰窖上传文件，或点击麦克风录制现场心声。")