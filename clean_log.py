import os
import re
import argparse

def clean_log_file(input_path, output_path):
    print(f"Start cleaning log file: {input_path}")
    
    if not os.path.exists(input_path):
        print(f"Error: File not found {input_path}")
        return

    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    
    total_lines = 0
    saved_lines = 0

    with open(input_path, 'r', encoding='utf-8', errors='ignore') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        
        for line in fin:
            total_lines += 1
            
            if "POST=" in line and "RECV=" in line:
                continue
                
            if "Progress:" in line or "s/case]" in line or "it/s]" in line:
                continue
            
            clean_line = ansi_escape.sub('', line)
            clean_line = clean_line.strip()
            
            if not clean_line:
                continue
            
            fout.write(clean_line + '\n')
            saved_lines += 1

    print("-" * 30)
    print("Clean Completed!")
    print(f"Total lines processed: {total_lines}")
    print(f"Valid lines saved: {saved_lines}")
    print(f"Garbage lines removed: {total_lines - saved_lines}")
    print(f"Cleaned log saved to: {output_path}")

if __name__ == "__main__":
    # 初始化 ArgumentParser
    parser = argparse.ArgumentParser(description="Clean dirty logs from aisbench.")
    
    # 添加 -s 和 -t 参数
    parser.add_argument("-s", "--source", required=True, help="Path to the source log file (e.g., all.log)")
    parser.add_argument("-t", "--target", required=True, help="Path to the target log file (e.g., clean.log)")
    
    # 解析参数
    args = parser.parse_args()
    
    # 传入函数执行
    clean_log_file(args.source, args.target)
