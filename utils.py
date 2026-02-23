import librosa
import librosa.display
import scipy.signal as signal
import numpy as np
import matplotlib.pyplot as plt 
import plotly.graph_objects as go
# ==========================================
# 【第一部分：核心后端算法】 
# 核心数值计算库
def process_audio_speed_by_temp(audio_series: np.ndarray, temperature: float) -> np.ndarray:
    
    # audio_series,temperature 为参数，后面的为类型注解
    """
    根据温度调整音频播放速度 (Time Stretch)。
    
    Args:
        audio_series (np.ndarray): 音频时间序列
        temperature (float): 温度值 (假设标准温度为 20.0)
        
    Returns:
        np.ndarray: 处理后的音频数据
    """
    # 1. 定义映射逻辑: 将温度映射为播放速率 (rate)
    # 假设 20度是 1.0倍速。每升高 10度，速度增加 10%
    # 公式: rate = 1.0 + (temp - 20) * 0.01
    rate = temperature
    if (rate == 1.0):
        return audio_series
    
    # 2. 边界保护: Librosa 要求 rate 必须 > 0
    # 我们设置一个最小速率 0.1 (非常慢) 和最大速率 5.0 (非常快)
    #np.clip 是numpy的数组裁剪函数，作用是将数
    rate = np.clip(rate, 0.1, 5.0)
    
    # 3. 调用 Librosa 库方法
    try:
        processed_data = librosa.effects.time_stretch(y = audio_series, rate = rate)
    #  time_stretch 改变速度但不改变音高
    except Exception as e:
        print(f"Librosa 处理出错: {e}")
        return audio_series
     #try异常处理,except捕获异常并处理
    return processed_data

def process_audio_pitch_by_temp(audio_series: np.ndarray, temperature: float) -> np.ndarray:
    """
    根据温度调整音频音高 (Pitch Shift)。
    温度越高，音调越高。
    """
    # 1. 定义映射逻辑: 将温度映射为半音 (n_steps)
    # 假设 20度是原调。
    # 温度每变化 5度，音高变化 1 个半音 (semitone)
    n_steps = (temperature - 1.0) * 10.0  #音高变化的半音步数
    if (temperature == 1.0):
        return audio_series
    
    # 2. 调用 Librosa 库方法
    # sr (采样率) 默认为 22050，这是 librosa 加载音频时的默认值。
    # 如果你的音频不是 22050Hz，这里的音高变化在时间轴上可能会有细微偏差，但不会报错。
    #pitch_shift函数实现音频音高调整
    try:
        processed_data = librosa.effects.pitch_shift(
            y=audio_series, 
            sr=22050, 
            n_steps=n_steps
        )
    except Exception as e:
        print(f"Librosa 处理出错: {e}")
        return audio_series
    return processed_data

def apply_lowpass_filter(audio_series: np.ndarray, sr: int, cutoff_freq: float) -> np.ndarray:
    """
    使用巴特沃斯低通滤波器 (Butterworth Low-pass Filter) 削弱刺耳的高频信号。
    
    Args:
        audio_series: 音频时间序列
        sr: 采样率
        cutoff_freq: 截止频率 (Hz)。高于此频率的信号将被大幅削弱。
    """
    # 1. 计算奈奎斯特频率 (Nyquist frequency)，理论上它是采样率的一半
    nyquist = 0.5 * sr
    
    # 2. 归一化截止频率 (要求值在 0.0 到 1.0 之间)
    normal_cutoff = cutoff_freq / nyquist
    
    # 防止截止频率设置过高导致归一化后溢出
    normal_cutoff = np.clip(normal_cutoff, 0.01, 0.99)
    
    # 3. 设计一个 2 阶的巴特沃斯滤波器 (阶数越高，过滤边缘越陡峭、越干净利落)
    b, a = signal.butter(2, normal_cutoff, btype='low', analog=False)
    
    # 4. 应用滤波器。使用 filtfilt 可以进行正反向两次滤波，保证波形不发生相位偏移
    filtered_audio = signal.filtfilt(b, a, audio_series)
    
    return filtered_audio

def apply_saturation(audio_series: np.ndarray, drive: float) -> np.ndarray:
    """
    【新增算法】软削峰饱和失真 (Soft Clipping Saturation)
    利用双曲正切函数 tanh，在不爆音的前提下，强行增加谐波能量，模拟"嘶吼感"。
    """
    # drive 越大，波形被挤压得越厉害，失真/炽热感越强
    return np.tanh(audio_series * drive)

# 新增：统一处理速度+音高的函数
def process_audio_speed_and_pitch(audio_series: np.ndarray, temperature: float, sr: int = 22050) -> np.ndarray:
    """
    统一音频处理总控函数：根据温度综合调整速度、音高与频域特征。
    注意：这里新增了 sr (采样率) 参数，因为滤波需要知道采样率。
    """
    current_audio = audio_series

    real_rate = 1.0 + (temperature - 1.0) * 0.25 
    real_pitch = (temperature - 1.0) * 1.5
    
    try:
        if real_rate != 1.0:
            current_audio = librosa.effects.time_stretch(y=current_audio, rate=real_rate)
        if real_pitch != 0.0:
            current_audio = librosa.effects.pitch_shift(y=current_audio, sr=sr, n_steps=real_pitch)
    except Exception as e:
        print(f"DSP引擎处理异常: {e}")
        return audio_series
    
    if temperature < 1.0:
        # 【暴力降温】：针对 Violent 音频
        # 温度 0.5 时，截止频率暴降至 1200Hz，强行抹除怒吼的共振峰
        # 温度 0.9 时，截止频率恢复到 6000Hz 左右
        target_cutoff = 1200 + (temperature - 0.5) * (9600)
        current_audio = apply_lowpass_filter(current_audio, sr, target_cutoff)
        
        # 稍微降低整体音量，配合柔软的听感
        current_audio = current_audio * (0.8 + (temperature - 0.5) * 0.4)

    elif temperature > 1.0:
        # 【烈火淬炼】：针对 Soft 音频
        # 计算过载驱动力 (Drive)：温度越高，Drive 越大 (最高可达 3.0)
        drive_amount = 1.0 + (temperature - 1.0) * 4.0
        
        # 施加饱和失真，无中生有地创造"愤怒的张力"，同时 tanh 天然防爆音
        current_audio = apply_saturation(current_audio, drive_amount)

    return current_audio
    
   
# ==========================================
# 【第二部分：绘图逻辑函数】 - 负责后端的“画图”动作
# ==========================================

def draw_waveform_plotly(y: np.ndarray, sr: int, title: str, color: str):
    """
    使用 Plotly 绘制交互式波形图 (自带降采样防御机制)
    """
    # 1. 内存防御：计算降采样步长 (Stride)
    max_points = 2000
    total_points = len(y)
    
    if total_points > max_points:
        step = total_points // max_points
        # 类似 C 语言的循环步长遍历，Python 的切片语法 y[start:stop:step]
        y_downsampled = y[::step] 
    else:
        y_downsampled = y
        
    # 生成对应的时间轴
    time_axis = np.linspace(0, total_points / sr, len(y_downsampled))
    
    # 2. 构建面向前端的轻量化图表对象
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=time_axis,
        y=y_downsampled,
        mode="lines",
        line=dict(color=color, width=1.5),
        name="Amplitude",
        hoverinfo="none" # 关闭鼠标悬停气泡，极限压榨渲染性能
    ))
    
    # 3. 剥离冗余 UI，设定绝对坐标系
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        xaxis_title="Time (s)",
        yaxis_title="Amplitude",
        height=220, # 配合侧边栏压低高度
        margin=dict(l=10, r=10, t=40, b=10), # 极简化边缘留白
        plot_bgcolor="rgba(0,0,0,0)", # 透明背景，融入网页主题
        paper_bgcolor="rgba(0,0,0,0)"
    )
    
    # 锁定 Y 轴振幅的物理边界，防止拖动温度时波形图忽大忽小
    fig.update_yaxes(range=[-1.0, 1.0])
    
    return fig

def draw_spectrogram(y, sr, title):
    """绘制声谱图,返回Matplotlib画布对象"""
    plt.style.use('default')  # 重置绘图样式
    fig, ax = plt.subplots(figsize=(10, 3))
    
    stft_result = librosa.stft(y)  #将音频从（幅度随时间变化）转换为（频率随时间变化）
    
    db_data = librosa.amplitude_to_db(np.abs(stft_result), ref=np.max)
    
    img = librosa.display.specshow(    # 绘制声谱图
        db_data,   
        sr=sr,
        x_axis='time',
        y_axis='hz',
        ax=ax,  #在ax子图上绘制
        cmap='viridis' #配色
    )
    ax.set_title(title, fontsize=12, pad=10)
    fig.colorbar(img, ax=ax, format="%+2.f dB", shrink=0.8) #添加颜色条，解释声谱图中颜色与分贝值的对应关系
    plt.tight_layout()
    return fig

# 首先，np.abs(stft_result) 计算短时傅里叶变换（STFT）结果的幅值谱，因为 STFT 结果是复数矩阵，幅值反映了每个时间和频率点上的能量强度。然后，librosa.amplitude_to_db 函数将这些幅值数据转换为分贝刻度。分贝是一种对数单位，更符合人耳对声音强度的感知。

# 参数 ref=np.max 表示以幅值中的最大值作为参考点，转换后的分贝值会以最大幅值为 0 dB，其他值相对于最大值进行缩放。这种处理方式有助于在可视化声谱图时突出能量的相对分布，使得弱信号和强信号都能清晰显示。

# 最终，db_data 就是可以直接用于声谱图绘制的分贝数据矩阵。