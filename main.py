# main.py
import streamlit as st
import librosa
import utils           # 导入我们的工具箱
import ui_components   # 导入我们的UI组件
import io
import datetime
import tempfile

# 使用装饰器，并添加一个友好的加载提示动画
@st.cache_data(show_spinner="⏳ 正在凝结底层冰晶 (解码音频)...")
def load_audio_from_bytes(audio_bytes):
    """
    将二进制音频流解码为 NumPy 数组。
    由于有 @st.cache_data 护航，相同的 audio_bytes 只会被解码一次。
    """
    # 使用 tempfile 在系统临时目录安全地创建一个无名文件，避免多线程冲突
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
        
    # 执行极其耗时的解码操作
    y, sr = librosa.load(tmp_path, sr=None)
    
    # 返回解码后的纯净数据
    return y, sr
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
if 'last_upload_id' not in st.session_state:
    st.session_state['last_upload_id'] = None


uploaded_file, recorded_audio_bytes = ui_components.render_sidebar_inputs()

# 3. 核心路由逻辑，及时更新后台金库 
if recorded_audio_bytes is not None and recorded_audio_bytes != st.session_state['last_record_bytes']:
    st.session_state['last_record_bytes'] = recorded_audio_bytes
    time_str = datetime.datetime.now().strftime("%H:%M:%S")
    new_name = f"即兴心声_{time_str}.wav"
    st.session_state['audio_vault'][new_name] = recorded_audio_bytes
    st.session_state['current_target'] = new_name

elif uploaded_file is not None:
    # 提取当前上传文件的唯一特征 (名称_大小)
    current_upload_id = f"{uploaded_file.name}_{uploaded_file.size}"
    
    # 核心拦截逻辑: 仅当特征改变时, 才说明是全新的物理上传动作
    if current_upload_id != st.session_state['last_upload_id']:
        st.session_state['last_upload_id'] = current_upload_id
        new_name = uploaded_file.name
        
        # 直接写入/覆盖金库
        st.session_state['audio_vault'][new_name] = uploaded_file.getvalue()
        st.session_state['current_target'] = new_name

# 4. 渲染侧边栏的历史记录组件 
selected_history, delete_triggered, files_to_delete = ui_components.render_sidebar_history()

# 补充场景：如果用户点击了历史记录按钮，切换游标
if selected_history is not None:
    st.session_state['current_target'] = selected_history
    
if delete_triggered and files_to_delete:
    for name in files_to_delete:
        # 1. 内存释放：从字典中彻底删除该键值对
        if name in st.session_state['audio_vault']:
            del st.session_state['audio_vault'][name]
        
        # 2. 游标安全校验：如果正在播放的文件被删了，必须将游标清空
        if st.session_state['current_target'] == name:
            st.session_state['current_target'] = None
    
    # 3. 强制页面重载：数据清理完毕后，立刻刷新前端画面
    st.rerun()

# 5. 渲染主界面并执行底层信号处理
ui_components.render_header()
target_name = st.session_state.get('current_target')

if target_name and target_name in st.session_state['audio_vault']:
    try:
        # 1. 拿到纯净的二进制数据
        raw_bytes = st.session_state['audio_vault'][target_name]
        
        # 2. 调用缓存函数！
        # 只要你还在处理同一个音频 (raw_bytes 没变)，滑动温度条时这里将瞬间执行完毕，耗时几乎为 0 毫秒！
        y, sr = load_audio_from_bytes(raw_bytes)
        
        st.markdown(f"**当前聆听:** `{target_name}`")
        
        # 3. 实时渲染控制区
        temperature = ui_components.render_controls()
        
        # 4. DSP 引擎处理 (这部分是纯内存 NumPy 矩阵运算，非常快，不需要缓存)
        y_processed = utils.process_audio_speed_and_pitch(y, temperature, sr)
        
        ui_components.render_tabs_content(y, y_processed, sr, temperature)
        
    except Exception as e:
        st.error(f"处理音频时遇到干扰: {e}")
else:
    st.info("👈 请在左侧拾遗冰窖上传文件，或点击麦克风录制现场心声。")