from asyncio import subprocess

from kokoro import KPipeline
import soundfile as sf
import numpy as np
import csv
import librosa
import pyrubberband as pyrb
import subprocess
import os

pipeline = KPipeline(lang_code='a')  

with open ("sentences_fromGdoc.csv",mode='r') as f:
    reader = csv.reader(f)
    text = list(reader)

    for a in text:
        print(a)
        #perform tts for both restrictive and non restrictive version. 
        for indx in (1,2):
            generator = pipeline(a[indx], voice='af_heart',speed=0.65)  
            
            # Collect all audio chunks
            audio_chunks = []
            for _, _, audio in generator:
                audio_chunks.append(audio)

            # Save to file with slowed talking speed 
            full_audio = np.concatenate(audio_chunks)

            
            #full_audio=librosa.effects.time_stretch(concat_audio, rate=0.65)
            #pitch_down=librosa.effects.pitch_shift(full_audio,n_steps=-3,sr=24000)

            tmpOUT="tmp_out.wav"
            id=a[0]
            if indx==1:
                pathTmp=f"{id}_r.wav"
                sf.write(pathTmp, full_audio, 24000)
                print("Saved output.wav")
                SPEED_RATE = 1
                TRIM_START = 0.5
                subprocess.run([
                "ffmpeg", "-y",
                "-i", pathTmp,
                "-filter:a", f"atempo={SPEED_RATE},atrim=start={TRIM_START},asetpts=PTS-STARTPTS",
                tmpOUT
                ], check=True)
            elif indx==2:
                pathTmp=f"{id}_n.wav"
                sf.write(pathTmp, full_audio, 24000)
                print("Saved output.wav")
                SPEED_RATE = 1
                TRIM_START = 0.5
                subprocess.run([
                "ffmpeg", "-y",
                "-i", pathTmp,
                "-filter:a", f"atempo={SPEED_RATE},atrim=start={TRIM_START},asetpts=PTS-STARTPTS",
                tmpOUT
                ], check=True)
            os.replace(tmpOUT, pathTmp)
            

            #now we need to perform forced alignment on the audio

       