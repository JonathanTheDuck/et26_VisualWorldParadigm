from kokoro import KPipeline
import soundfile as sf
import numpy as np
import csv


pipeline = KPipeline(lang_code='a')  

with open ("sentences.csv",mode='r') as f:
    reader = csv.reader(f)
    text = list(reader)

    for a in text:
        print(a)
        generator = pipeline(a[1], voice='af_heart',speed=0.7)  
        
        # Collect all audio chunks
        audio_chunks = []
        for _, _, audio in generator:
            audio_chunks.append(audio)

        # Save to file
        full_audio = np.concatenate(audio_chunks)
        id=a[0]
        sf.write(f"{id}_r.wav", full_audio, 24000)
        print("Saved output.wav")

        generator_n = pipeline(a[2], voice='af_heart',speed=0.7) 
        # Collect all audio chunks
        audio_chunks_n = []
        for _, _, audio in generator_n:
            audio_chunks_n.append(audio)

        # Save to file
        full_audio_n = np.concatenate(audio_chunks_n)
        sf.write(f"{id}_n.wav", full_audio_n, 24000)
        print("Saved output.wav")