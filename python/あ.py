    # Gemini API処理とコプター動作、録画録音を並列処理
    gemini_process = multiprocessing.Process(target=Gemini, args=(shared_data,))
    log_move_process =  multiprocessing.Process(target=log_move, args=(shared_data, base_timestamp, current_record_folder))
    record_process = multiprocessing.Process(target=record, args=(duration, shared_data))
    
    # 1回目の録音録画を実行
    record_process.start()
    record_process.join()
    print("video and audio path: ", shared_data["last_video_path"], shared_data["last_audio_path"])
    

    # 2回目の録音録画と1回目のGemini API処理を実行
    
    gemini_process.start()
    record_process.start()
    gemini_process.join()
    record_process.join()

    print("video and audio path: ", shared_data["last_video_path"], shared_data["last_audio_path"])
    print("Gemini Response: ", shared_data["gemini_response"])


    while True:
        try:
            # 1回目のコプター動作と2回目のGemini API処理と3回目の録画録音を実行
            log_move_process.start()
            gemini_process.start()
            record_process.start()
            
            log_move_process.join()
            gemini_process.join()
            record_process.join()
            
        except Exception as e:
            print(f"メインループエラー: {e}")
            time.sleep(5) 