import sys
import json
import wave
import os
from vosk import Model, KaldiRecognizer

def recognize_audio(audio_file, model_path="model"):
    """
    使用 Vosk 识别音频文件
    """
    # 检查模型是否存在
    if not os.path.exists(model_path):
        print(f"错误：模型文件夹 '{model_path}' 不存在")
        print("请从 https://alphacephei.com/vosk/models 下载模型")
        print("推荐下载 vosk-model-small-cn-0.22，解压后重命名为 'model'")
        return None
    
    # 加载模型
    print("正在加载模型...")
    try:
        model = Model(model_path)
        print("模型加载成功")
    except Exception as e:
        print(f"模型加载失败: {e}")
        return None
    
    # 检查音频文件
    if not os.path.exists(audio_file):
        print(f"错误：音频文件 '{audio_file}' 不存在")
        return None
    
    # 打开音频文件
    try:
        wf = wave.open(audio_file, "rb")
    except Exception as e:
        print(f"无法打开音频文件: {e}")
        print("请确保文件格式为 WAV 格式")
        return None
    
    # 创建识别器
    rec = KaldiRecognizer(model, wf.getframerate())
    rec.SetWords(True)
    
    print("正在识别音频...")
    results = []
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            result = json.loads(rec.Result())
            text = result.get("text", "")
            if text:
                results.append(text)
                print(f"已识别片段: {text}")
    
    # 获取最终结果
    final_result = json.loads(rec.FinalResult())
    final_text = final_result.get("text", "")
    if final_text:
        results.append(final_text)
    
    return " ".join(results)

def main():
    if len(sys.argv) < 2:
        print("用法: python asr_demo.py <音频文件路径>")
        print("示例: python asr_demo.py voice_clone.wav")
        print("\n注意：Vosk 支持 WAV 格式。如果是 MP3 文件，请先用格式转换工具转为 WAV")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    print(f"正在识别: {audio_file}")
    
    text = recognize_audio(audio_file)
    
    if text:
        print("\n" + "=" * 50)
        print("最终识别结果：")
        print("=" * 50)
        print(text)
    else:
        print("识别失败")

if __name__ == "__main__":
    main()
