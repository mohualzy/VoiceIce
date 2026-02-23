# main.py
import streamlit as st
import librosa
import utils           # 导入我们的工具箱
import ui_components   # 导入我们的UI组件
import io

# 1. 页面设置
st.set_page_config(page_title="言冰 Voiceice", page_icon="🧊", layout="wide")

# 2. 初始化 
if 'history' not in st.session_state:
    st.session_state['history'] = []

# 3. 渲染侧边栏，并获取上传的文件
uploaded_file, recorded_audio_bytes = ui_components.render_sidebar()

# 4. 渲染主标题
ui_components.render_header()

# 5. 核心逻辑
audio_source = None
file_name_for_history = ""

if recorded_audio_bytes is not None:
    # 如果用户按了麦克风录音
    audio_source = io.BytesIO(recorded_audio_bytes)
    file_name_for_history = "现场即兴录音"
elif uploaded_file is not None:
    # 如果用户上传了文件
    audio_source = uploaded_file
    file_name_for_history = uploaded_file.name

# 如果有音频来源，就开始处理
if audio_source is not None:
    try:
        # librosa 可以直接读取 BytesIO 对象
        y, sr = librosa.load(audio_source, sr=None)
        
        temperature = ui_components.render_controls()
        
        y_processed = utils.process_audio_speed_and_pitch(y, temperature, sr)
        
        current_record = {'name': file_name_for_history, 'temp': temperature}
        
        st.session_state['history'] = [
            rec for rec in st.session_state['history']
            if not (rec['name'] == current_record['name'] and rec['temp'] == current_record['temp'])
        ]
        
        st.session_state['history'].insert(0, current_record)
            
        ui_components.render_tabs_content(y, y_processed, sr, temperature)
        
    except Exception as e:
        st.error(f"处理音频时遇到干扰: {e}")

else:
    st.info("👈 请在左侧【拾遗冰窖】上传文件，或使用【现场采音】功能开始体验。")