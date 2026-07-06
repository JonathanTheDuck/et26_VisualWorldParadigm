#sourced from sample on github https://github.com/MahmoudAshraf97/ctc-forced-aligner

import torch
import csv
import spacy
from ctc_forced_aligner import (
    load_audio,
    load_alignment_model,
    generate_emissions,
    preprocess_text,
    get_alignments,
    get_spans,
    postprocess_results,
)

def align(tex="The boy will move the cake", audio_path="0_n.wav", outList=None):

    language = "iso" # ISO-639-3 Language code
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 16


    alignment_model, alignment_tokenizer = load_alignment_model(
    device,
    dtype=torch.float16 if device == "cuda" else torch.float32,
    )

    audio_waveform = load_audio(audio_path, alignment_model.dtype, alignment_model.device)

    emissions, stride = generate_emissions(
        alignment_model, audio_waveform, batch_size=batch_size
    )

    tokens_starred, text_starred = preprocess_text(
        tex,
        romanize=True,
        language=language,
    )

    segments, scores, blank_token = get_alignments(
        emissions,
        tokens_starred,
        alignment_tokenizer,
    )

    spans = get_spans(tokens_starred, segments, blank_token)

    word_timestamps = postprocess_results(text_starred, spans, stride, scores)
    outList.append(word_timestamps)

list_ofAlignments=[]

with open ("creatingDataStructure/sentences.csv",mode='r') as f:
    reader = csv.reader(f)
    text = list(reader)
    for a in text:
        print(a)
        #perform tts for both restrictive and non restrictive version. 
        for indx in (1,2):
            if indx==1:
                audio_path=f"creatingDataStructure/{a[0]}_r.wav"
            elif indx==2:
                audio_path=f"creatingDataStructure/{a[0]}_n.wav"
            align(tex=a[1],audio_path=audio_path, outList=list_ofAlignments)


