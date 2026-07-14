#sourced from sample on github https://github.com/MahmoudAshraf97/ctc-forced-aligner

import json
from collections import defaultdict
import torch
import csv
import spacy
import pandas as pd
from ctc_forced_aligner import (
    load_audio,
    load_alignment_model,
    generate_emissions,
    preprocess_text,
    get_alignments,
    get_spans,
    postprocess_results,
)

#list_ofAlignments=defaultdict(dict)

annotation_rows=[]
def append_allignment(id, sentenceRole,wordRole, word,start,end, sentence):
    annotation_rows.append({
        "id": id,
        "word": word,
        "SentenceRole": sentenceRole,
        "WordRole": wordRole,
        "start": start,
        "end": end,
        "sentence": sentence
    })


language = "iso" # ISO-639-3 Language code
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 16


alignment_model, alignment_tokenizer = load_alignment_model(
device,
dtype=torch.float16 if device == "cuda" else torch.float32,
)

#load pos tagger in addition to the timestamp model
nlp = spacy.load("en_core_web_sm")

def align(id,tex, audio_path, role=None):


    #load audio as wave
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
    print(word_timestamps)
    #also determine roles of the words via spacy 
    doc=nlp(tex)
    roles = [(token.text, token.dep_) for token in doc]
    roleDict=defaultdict(str)
    for word, word_role in roles:
        roleDict[word] = word_role
    print(roleDict)
    
    for word_package in word_timestamps:
        append_allignment(id=id, word=word_package["text"], sentenceRole=role, wordRole=roleDict[word_package["text"]], start=word_package["start"], end=word_package["end"], sentence=tex)
    print(append_allignment)


with open ("creatingDataStructure/sentences_fromGdoc.csv",mode='r') as f:
    reader = csv.reader(f)
    text = list(reader)[1:]
    for a in text:
        print(a)
        #perform tts for both restrictive and non restrictive version. 
        for indx in (1,2):
            if indx==1:
                audio_path=f"audio_v02/{a[0]}_r.wav"
                role="restrictive"
            elif indx==2:
                audio_path=f"audio_v02/{a[0]}_n.wav"
                role="non-restrictive"
            align(id=a[0], tex=a[indx], audio_path=audio_path, role=role)
    annotation_df=pd.DataFrame(annotation_rows, columns=["id","word","SentenceRole","WordRole","start","end","sentence"])
    annotation_df.to_csv( "annotation.csv", index=False)


