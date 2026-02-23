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
if 'history' not in st.session_state:
    st.session_state['history'] = []

# 3. 渲染侧边栏，并获取上传的文件
uploaded_file, recorded_audio_bytes = ui_components.render_sidebar()

# 4. 渲染主标题
ui_components.render_header()

# 5. 核心逻辑
audio_source = None
file_name_for_history = ""

if 'last_record_bytes' not in st.session_state:
    st.session_state['last_record_bytes'] = None
if 'last_record_name' not in st.session_state:
    st.session_state['last_record_name'] = ""

if recorded_audio_bytes is not None:
    # 只有当录音内容发生物理变化时，才生成新的时间戳文件名
    if recorded_audio_bytes != st.session_state['last_record_bytes']:
        st.session_state['last_record_bytes'] = recorded_audio_bytes
        time_str = datetime.datetime.now().strftime("%H:%M:%S")
        st.session_state['last_record_name'] = f"即兴心声_{time_str}.wav"
    
    file_name_for_history = st.session_state['last_record_name']
    
    # 将内存字节流写入物理文件，避开 librosa 的内存解码陷阱
    temp_path = "temp_live_record.wav"
    with open(temp_path, "wb") as f:
        f.write(recorded_audio_bytes)
    audio_source = temp_path

elif uploaded_file is not None:
    audio_source = uploaded_file
    file_name_for_history = uploaded_file.name

# 执行处理流水线
if audio_source is not None:
    try:
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
    st.info("👈 请在左侧拾遗冰窖上传文件，或点击麦克风录制现场心声。")