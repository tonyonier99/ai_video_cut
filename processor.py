import json
import os
import sys
from moviepy import VideoFileClip

def process_video(video_path, json_path):
    # Check if files exist
    if not os.path.exists(video_path):
        print(f"錯誤：找不到影片檔案 {video_path}")
        return
    if not os.path.exists(json_path):
        print(f"錯誤：找不到裁切紀錄檔 {json_path}")
        return

    # Load targets
    with open(json_path, 'r', encoding='utf-8') as f:
        cuts = json.load(f)

    print(f"🎥 開始處理影片：{os.path.basename(video_path)}")
    print(f"📦 預計產出 {len(cuts)} 個精華片段...")

    # Create output directory
    output_dir = "exports"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        video = VideoFileClip(video_path)
        
        for i, cut in enumerate(cuts):
            start_t = cut['start']
            end_t = cut['end']
            label = cut.get('label', f'clip_{i}')
            
            # Simple filename sanitization
            safe_label = "".join([c for c in label if c.isalnum() or c in (' ', '_')]).rstrip()
            output_filename = f"{output_dir}/Highlight_{i+1}_{safe_label}.mp4"
            
            print(f"🎬 正在剪輯片段 {i+1}: {start_t}s -> {end_t}s...")
            
            # Subclip and write
            new_clip = video.subclipped(start_t, end_t)
            new_clip.write_videofile(output_filename, codec="libx264", audio_codec="aac")
            
        video.close()
        print(f"\n✨ 全部完成！影片已儲存在：{os.path.abspath(output_dir)}")
        
    except Exception as e:
        print(f"❌ 剪輯過程中發生錯誤：{str(e)}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("使用方法: python processor.py <影片路徑> <JSON路徑>")
    else:
        process_video(sys.argv[1], sys.argv[2])
