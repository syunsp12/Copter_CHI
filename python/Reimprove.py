import os
import time
import json
import datetime
from datetime import datetime
import threading
import multiprocessing

import cv2
import requests
import base64
import pyaudio
import wave
from pydub import AudioSegment
from buildhat import Motor

import vertexai
import vertexai.generative_models as genai

# 一時保存フォルダのパス設定と作成
temp_save_folder = "/home/pi/Desktop/temp"
temp_video_folder = os.path.join(temp_save_folder, "Video")
temp_audio_folder = os.path.join(temp_save_folder, "Audio")

# ファイルがない場合新しく作成
if not os.path.exists(temp_video_folder):
    os.makedirs(temp_video_folder)
if not os.path.exists(temp_audio_folder):
    os.makedirs(temp_audio_folder)

# ログ保存用のフォルダパスを設定
record_save_folder = "/home/pi/Desktop/Record"

# Vertex AI APIの設定
PROJECT_ID = "792391311320"
REGION = "asia-northeast1" 
vertexai.init(project=PROJECT_ID, location=REGION)

# モーターの初期化（ポートを指定）
motor = Motor("A")

# モーターのリリースを有効化（停止後にモーターが固定されるように）
motor.release = False

# この行でstderrを/dev/nullにリダイレクト (alsaのログを減らす)
os.dup2(os.open(os.devnull, os.O_RDWR), 2)

# カメラを初期化（デフォルトカメラ）
cap = cv2.VideoCapture(0)
if cap.isOpened():
    print("カメラの初期化が完了しました。")
else:
    raise RuntimeError("カメラの初期化に失敗しました。")

# カメラの解像度の指定
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# サブプロセスの応答を格納する変数
manager = multiprocessing.Manager()
gemini_response = manager.dict()
motor_response = manager.dict()

NotRecord_path = "path"

def record_video_audio(duration, sample_rate=44100, chunk_size=1024):
    """
    指定された秒数間、カメラで録画を行い、音声を同時に録音します。
    終了後、ビデオとオーディオの保存されたファイルパスを返します。
    """
    try:
        # タイムスタンプ付きのファイル名を生成
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_path = os.path.join(temp_video_folder, f"temp_Video_{timestamp}.mp4")
        audio_path = os.path.join(temp_audio_folder, f"temp_mp3_audio_{timestamp}.mp3")
        temp_wav_path = os.path.join(temp_audio_folder, f"temp_wav_audio_{timestamp}.wav")
        
        # 録画の設定 (FourCC形式とフレームレートを指定)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        output = cv2.VideoWriter(video_path, fourcc, 1.0, (640, 480)) 
        
        # ビデオライターが開けない場合のエラーハンドリング
        if not output.isOpened():
            print("ビデオライターが開けませんでした。")
            cap.release()
            return None, None
            
        # PyAudioオブジェクトの初期化
        p = pyaudio.PyAudio()
        audio_format = pyaudio.paInt16  # 16ビット音声
        channels = 1  # モノラル録音

        # 音声ストリームのオープン
        stream = p.open(format=audio_format, channels=channels, rate=sample_rate, input=True, frames_per_buffer=chunk_size)

        print("録画と録音を開始します。")
        frames = []

        start_time = time.time()

        # 指定された秒数間、フレームと音声データをキャプチャして保存
        while (time.time() - start_time) < duration: 
            ret, frame = cap.read()
            
            if not ret:
                print("フレームの読み込みに失敗しました。")
                break
            
            output.write(frame)
            cv2.imshow('Recording', frame)  # フレームをリアルタイムで表示
            cv2.waitKey(1)

            data = stream.read(chunk_size)
            frames.append(data)

        print("録画と録音を終了します。")

        # ストリームの停止とクローズ
        stream.stop_stream()
        stream.close()
        p.terminate()

        # 録画の終了処理
        output.release()
        cv2.destroyAllWindows()

        # 一時的にWAVファイルとして保存
        with wave.open(temp_wav_path, "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(p.get_sample_size(audio_format))
            wf.setframerate(sample_rate)
            wf.writeframes(b"".join(frames))
            
        # 一時的にWAVファイルとして保存した後、音量を調整
        audio = AudioSegment.from_wav(temp_wav_path)

        # 全体の音量を一括で6dB上げる
        audio = audio + 6

        # WAVファイルをMP3に変換して保存
        audio.export(audio_path, format="mp3")

        # 一時的なWAVファイルを削除
        os.remove(temp_wav_path)
        
    except Exception as e:
        print(f"録画・録音エラー: {e}")
        return None, None

    return video_path, audio_path
    """
    録画と録音を並行して実行します。
    終了後、ビデオとオーディオの保存パスを返します。
    """
    try:
        result_list = []

        # 録音処理を別スレッドで実行
        audio_thread = threading.Thread(target=record_audio, args=(duration, 44100, 1024, result_list))

        audio_thread.start() # 録音開始
        record_video(duration, result_list) # メインスレッドで録画実行
        audio_thread.join() # 録音スレッドの終了を待機

        video_path = result_list[0] if len(result_list) > 0 else None
        audio_path = result_list[1] if len(result_list) > 1 else None
        
    except Exception as e:
        print(f"録画・録音の同時処理エラー: {e}")
        return None, None

    return video_path, audio_path

def upload_and_prompt_video_audio(video_path, audio_path):
    """
    録画されたビデオファイルと録音されたオーディオファイルをGeminiにアップロードし、
    指定されたプロンプトを入力して応答を得ます。
    """
    try:
        video_part = genai.Part.from_data(open(video_path, "rb").read(), mime_type="video/mp4")
        audio_part = genai.Part.from_data(open(audio_path, "rb").read(), mime_type="audio/mp3")

        prompt = """
             # 指示
            あなたはraspberrypi上にモーターが接続され、モーターに可動部が接続されたプロダクト「コプター」です。
            あなたの性格をもとに、入力された環境下において動くコプターの動作パターンを生成してください。
            ※コプターはモーターに接続されているため回転以外の動作は行いません。そのためあなたの意図が、適切に回転動作に反映されるように工夫して動作パターンを出力してください。

            # 入力
            コプターが見ている周囲の環境を示したmp4の動画データ
            コプターが聞いている周囲の環境を示したmp3の音声データ
            ※これら2つの入力データは同時に取得が開始され、タイムスタンプが一致しています。

            # 出力
            JSON形式のデータを出力してください。
            モーターの動作ごとに繰り返し出力されます。
            モーターの動作回数は環境に応じて変動します。
            JSON規格に従い、コメントを含めないでください。
            コプターの動作を表現してください。

            ## JSON形式の詳細
            - "interpretation"は3つのパターンに分類されます:
                - "interpretation_Vison": コプターが見ている映像から環境の状況を客観的に表現
                - "interpretation_Audio": コプターが聞いている音声から環境の状況を客観的に表現
                - "interpretation_Think": "interpretation_Vison"と"interpretation_Audio"を基にコプターが何を思考してその動作に至ったのかを表現してください

            - "motorMovements"は２つの"type"があり、モーターの動作を示します:
                - "type": "run_for_degrees"では回転角度"angle"(-360~360)と回転速度"speed"(0~100)を指定
                - "type": "sleep"では停止秒数"second"(0.0~60.0)を指定

            - 出力は以下の順序で行う必要があります：
            1. "interpretation_Vison"
            2. "interpretation_Audio"
            3. "interpretation_Think"
            4. "motorMovements"
            
            -出力にあたっては以下の制約を必ず遵守してください
            - "interpretation_Vison", "interpretation_Audio", "interpretation_Think"は出力のはじめに一度のみ記述されます。一回の記述の中で時系列順で過程を表現してください。
            - モーターは"angle"360,"speed"100で約1秒で回転します。これを加味してコプターは”run_for_degrees”している時間と"sleep"している時間を合わせて0~60秒間動作するように動作パターンを出力してください
            - コプターの動作に当たっては”run_for_degrees”を必ず1度は出力すること
            # 出力例
            {
            "interpretation_Vison": "森の中で撮影されており、木々が生い茂っている。遠くに小川が流れている。",
            "interpretation_Audio": "川のせせらぎ音と鳥のさえずりが聞こえる。",
            "interpretation_Think": "この環境はとても興味深い。先に何があるのか探検したくなってきた。小川の源泉を見つける冒険に出よう。",
            "motorMovements": [
                {
                "type": "run_for_degrees",
                "angle": 180,
                "speed": 75
                },
                {
                "type": "sleep",
                "second": 2.0
                }...
            ]
            }

            #コプターの性格
            1. **効率的**
                - 車輪型のデザインから、このエージェントは非常に効率的でタスクを迅速にこなす性格を持っています。時間管理や効率性を重視し、最短距離で目的を達成することを目指します。

            2. **実用的**
                - シンプルで機能的な外見から、実用的で現実的な性格が伺えます。日常生活における実際のニーズを的確に捉え、使用者にとって最も役立つサポートを行います。

            3. **地道な努力家**
                - このエージェントは、地道に努力を続けるタイプです。継続的に努力し、目標を達成するために必要な過程を大切にします。。    
        """

        model = genai.GenerativeModel(model_name="gemini-1.5-pro", generation_config={"response_mime_type": "application/json"})

        print("Geminiの応答を待っています...")
        response = model.generate_content([video_part, audio_part, prompt])

        print(response.text)

    except Exception as e:
        print(f"Gemini APIのエラー: {e}")
        return None
    return response.text

def process_and_execute_response(response_text, base_timestamp, current_record_folder):
    """
    レスポンスの処理、モーターの動作、応答の記録、ファイルの移動を行う関数
    """
    try:
        response_json = json.loads(response_text)
        
        motor_movements = response_json.get("motorMovements", [])

        for movement in motor_movements:
            if "angle" in movement:
                movement["angle"] = int(movement["angle"])
            if "speed" in movement:
                movement["speed"] = int(movement["speed"])
            if "second" in movement:
                movement["second"] = float(movement["second"])

        interpretation_Vison = response_json.get("interpretation_Vison", "")
        interpretation_Audio = response_json.get("interpretation_Audio", "")
        interpretation_Think = response_json.get("interpretation_Think", "")
        
        log_file_path = os.path.join(current_record_folder, f"log_{base_timestamp}.txt")
        with open(log_file_path, "a") as log_file:
            log_file.write(f"\nTime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            log_file.write(f"Interpretation Vision: {interpretation_Vison}\n")
            log_file.write(f"Interpretation Audio: {interpretation_Audio}\n")
            log_file.write(f"Interpretation Think: {interpretation_Think}\n")
            log_file.write(f"Motor Movements: {motor_movements}\n")
            log_file.write("\n")

        for movement in motor_movements:
            type = movement["type"]
            if type == "run_for_degrees":
                angle = movement["angle"]
                speed = movement["speed"]
                if speed != 0:
                    motor.run_for_degrees(angle, speed)
                    print(f"Motor moved {angle} degrees at speed {speed}")
                else:
                    print(f"Speed is zero for angle {angle}, skipping...")
            elif type == "sleep":
                second = movement["second"]
                time.sleep(second)

        iteration_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_video_path = os.path.join(current_record_folder, f"video_{iteration_timestamp}.mp4")
        final_audio_path = os.path.join(current_record_folder, f"audio_{iteration_timestamp}.mp3")
        
        return final_video_path, final_audio_path

    except (KeyError, ValueError, IndexError, json.JSONDecodeError) as e:
        print(f"Error processing response: {e}")
        return None, None

def main():
    base_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    current_record_folder = os.path.join(record_save_folder, base_timestamp)
    os.makedirs(current_record_folder, exist_ok=True)

    while True:
        try:
            video_path, audio_path = record_video_audio_concurrently(duration=15)
            if video_path is None or audio_path is None:
                raise RuntimeError("録画または録音に失敗しました。")

            response_text = upload_and_prompt_video_audio(video_path, audio_path)
            if response_text is None:
                raise RuntimeError("Gemini APIの呼び出しに失敗しました。")

            final_video_path, final_audio_path = process_and_execute_response(response_text, base_timestamp, current_record_folder)

            if not final_video_path or not final_audio_path:
                continue

            os.rename(video_path, final_video_path)
            os.rename(audio_path, final_audio_path)


        except Exception as e:
            print(f"メインループエラー: {e}")
            time.sleep(10)  # エラー発生時に少し待ってから次を試行

if __name__ == "__main__":
    main()