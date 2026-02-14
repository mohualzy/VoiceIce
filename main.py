# main.py
import streamlit as st
import librosa
import utils           # 导入我们的工具箱
import ui_components   # 导入我们的UI组件

# 1. 页面设置
st.set_page_config(page_title="言冰 Voiceice", page_icon="🧊", layout="wide")

# 2. 初始化 
if 'history' not in st.session_state:
    st.session_state['history'] = []

# 3. 渲染侧边栏，并获取上传的文件
uploaded_file = ui_components.render_sidebar()

# 4. 渲染主标题
ui_components.render_header()

# 5. 核心逻辑
if uploaded_file is not None:
    try:
        # 加载音频 (后端逻辑)
        y, sr = librosa.load(uploaded_file, sr=None)
        
        # 渲染控制栏，并获取用户设定的温度 (前端交互)
        temperature = ui_components.render_controls()
        
        # 调用utils里的算法处理函数
        y_processed = utils.process_audio_speed_and_pitch(y, temperature)
        
        # 构建当前记录对象
        current_record = {'name': uploaded_file.name, 'temp': temperature}
        
        # 与AI对话侧边栏一样的更新置顶功能，并清除重复处理的记录
        st.session_state['history'] = [
            rec for rec in st.session_state['history']
            if not (rec['name'] == current_record['name'] and rec['temp'] == current_record['temp'])
        ]
        
        # 将当前最新的记录强行插入到列表的第 0 个位置（即最顶端）。
        st.session_state['history'].insert(0, current_record)
            
        # 渲染底部的所有图表
        ui_components.render_tabs_content(y, y_processed, sr, temperature)
        
    except Exception as e:
        print("再试一试呢?")

else:
    st.info("👈 请在左侧侧边栏【拾遗冰窖】中上传录音文件，开始体验。")