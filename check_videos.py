import os

mp4_files = [f for f in os.listdir() if f.endswith('.mp4')]

if mp4_files:
    print("Found .mp4 files:")
    for f in mp4_files:
        print(" -", f)
else:
    print("No .mp4 files found in the current directory.")
