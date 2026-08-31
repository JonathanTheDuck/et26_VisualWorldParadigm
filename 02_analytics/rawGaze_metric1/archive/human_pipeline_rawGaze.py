import pandas as pd
#importing all relevant files :)
#pls run this script from within the dir human-predictions ^^
partID=[]
out_stimuliId=[]
out_obj=[]
is_target=[]
out_count_per_obj_per_part=[]
out_percent=[]
condition=[]
for participantNumber in range(1,24):

    #collect everything in one dataframe for ease of use with llm based percentages and aggregation
    

    subjectnr=participantNumber
    groupNr=subjectnr %8+1

    path="../../01_experiment/recordings/SUBJECTS (FINAL)/"
    path_info="../../01_experiment/stimuli/creatingDataStructure/participant_groups_8/participant_group"+str(groupNr)+".csv"
    part_df_tmp=pd.read_csv(path+"subject-"+str(subjectnr)+".csv")
    gazeSamples_df=pd.read_table(path+"subject-"+str(subjectnr)+".tsv")
    gazeSamples_df_reduced=gazeSamples_df[["TIME","BPOGX","BPOGY","BPOGV","USER"]]
    gazeSamples_df.columns

    group_df=pd.read_csv(path_info)

    annotationPath="../../01_experiment/stimuli/annotation_audiov2.csv"
    annotation_df=pd.read_csv(annotationPath)

    part_df=part_df_tmp[["id","position1","position2","position3","position4","count_audio_file_offset","audio_file","condition","target_position"]]
    part_df.rename(columns={'count_audio_file_offset': 'trialID'}, inplace=True)
    part_df.rename(columns={'id': 'stimuliID'}, inplace=True)
    part_df

    #adding trial ids
    is_start = gazeSamples_df_reduced["USER"].eq("START_TRIAL")
    is_stop = gazeSamples_df_reduced["USER"].eq("AUDIO_FILE_OFFSET") #because the stop trial seems to have sometimes not been recorded correctly. 

    inside = False
    current_id = -1
    ids = []

    for start, stop in zip(is_start, is_stop):
        if start:
            inside = True
            current_id += 1
        #using str here to make merging easier later
        ids.append(str(current_id) if inside else None)

        if stop:
            inside = False

    gazeSamples_df_reduced["trialId"] = ids

    # Alles außerhalb eines Intervalls entfernen
    samples_onlyTrials = gazeSamples_df_reduced[gazeSamples_df_reduced["trialId"].notna()]
    samples_onlyTrials

    positions = {
        "pos1": (1891, 429),
        "pos2": (1423,915),
        "pos3": (697, 915),
        "pos4": (229, 429),
    }

    #also scaled these so they are not sqares anymore actually...need to take care of rsolution better next time !
    BOX_WIDTH = 440
    BOX_HEIGTH = 440
    #________________________________________

    #also it seems the screen resolution was at 1920x1080, if i read it coirrectly fromm the logfile. so we should change this for the real trials!
    #for now we need to convert image coords to screen coords that we have in open gaze. 



    #additional sanity check //outline created with claude:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from PIL import Image
    import numpy as np
    from matplotlib.patches import Rectangle

    # ── Config — adjust these ──────────────────────────────────────────
    #SCREEN_W, SCREEN_H = 1920, 1080
    IMG_PATH = "../../01_experiment/stimuli/img_composition/1_pos1_nosub.png"   # the actual displayed canvas image
    SCREEN_W, SCREEN_H = 2560, 1440  # OpenSesame virtual canvas
    IMG_W, IMG_H = 2560, 1440        # from your image

    SCALE = 1  # explicitly set in OpenSesame sketchpad

    disp_w = IMG_W * SCALE    # 1920.0
    disp_h = IMG_H * SCALE    # 1120.0
    offset_x = (SCREEN_W - disp_w) / 2  # 320.0
    offset_y = (SCREEN_H - disp_h) / 2  # 160.0

    # ── Convert gaze samples: normalized -> screen pixels ──────────────
    valid = samples_onlyTrials[samples_onlyTrials["BPOGV"] == 1].copy()
    valid["screen_x"] = valid["BPOGX"] * SCREEN_W
    valid["screen_y"] = valid["BPOGY"] * SCREEN_H

    # ── Plot: full screen canvas, image placed at its real position, gaze on top
    fig, ax = plt.subplots(figsize=(12, 6.75))  # matches 16:9 aspect

    # black screen background
    ax.add_patch(patches.Rectangle((0, 0), SCREEN_W, SCREEN_H, color="black", zorder=0))

    # image placed at its actual displayed position/size
    img = np.array(Image.open(IMG_PATH))
    ax.imshow(img, extent=[offset_x, offset_x + disp_w, offset_y + disp_h, offset_y], zorder=1)
    # note: extent y is [top, bottom] reversed because imshow origin is upper-left by default

    # gaze samples on top
    ax.scatter(valid["screen_x"], valid["screen_y"], s=10, c="red", alpha=0.5, zorder=2, label="gaze samples")

    for name, (x, y) in positions.items():
        rect = Rectangle(
            (x, y),            # top-left corner
            BOX_WIDTH,         # width
            BOX_HEIGTH,        # height
            linewidth=3,
            edgecolor='blue',
            facecolor='none'
        )
        ax.add_patch(rect)

    ax.set_xlim(0, SCREEN_W)
    ax.set_ylim(SCREEN_H, 0)  # y-axis flipped so (0,0) is top-left, matching screen coords
    ax.set_aspect("equal")
    ax.set_title(f"{len(valid)} valid gaze samples on screen ({SCREEN_W}x{SCREEN_H})")
    ax.legend(loc="upper right")
    plt.tight_layout()

    #(for sanity checking. even better if you check for all actual stimuli img instead of one. boundiing boxes will remain the same though)
    #plt.show()
    

    onsetsetsAudioFiles=samples_onlyTrials[samples_onlyTrials["USER"]=="AUDIO_FILE_ONSET_LOG"]
    onsetsetsAudioFiles.rename(columns={"TIME":"audio_onset_time"},inplace=True)
    onsetsetsAudioFiles=onsetsetsAudioFiles[["audio_onset_time","trialId"]]

    annotation_df.rename(columns={"id":"stimuliID"},inplace=True)
    annotation_df.rename(columns={"SentenceRole":"condition"},inplace=True)

    def assign_aoi(df, positions, box_width, box_height,
                x_col="screen_x", y_col="screen_y"):
        
        df = df.copy()
        df["AOI"] = None   # or pd.NA

        for name, (x0, y0) in positions.items():
            mask = (
                (df[x_col] >= x0) &
                (df[x_col] <= x0 + box_width) &
                (df[y_col] >= y0) &
                (df[y_col] <= y0 + box_height)
            )
            df.loc[mask, "AOI"] = name

        return df.dropna(subset=["AOI"]).reset_index(drop=True)


    for trialNr in range(0,50):

        #first we get all dataframes of one trial
        #print(trialNr)
        trial_info=part_df[part_df["trialID"]==trialNr]
        stimuliID=trial_info["stimuliID"].values[0]
        condition_trial=trial_info["condition"].values[0]
        
        audioStart_trial=onsetsetsAudioFiles[onsetsetsAudioFiles["trialId"]==str(trialNr)]["audio_onset_time"].values[0]

        #start and ending time:
        trial_annotation_df=annotation_df[annotation_df["stimuliID"]==stimuliID]
        trial_annotation_df=trial_annotation_df[trial_annotation_df["condition"]==condition_trial]
        #verb
        verbOnset=trial_annotation_df[trial_annotation_df["WordRole"]=="ROOT"]["start"].values[0]
        #object
        try:
            #here we found an error because w e only used dobj as role for the target objectz but üobj can also occur (actually only once)
            objectOnset=trial_annotation_df[(trial_annotation_df["WordRole"]=="dobj") | (trial_annotation_df["WordRole"]=="pobj")]["start"].values[0]
        except:
            print("error on"+str(stimuliID))
        
        
        #petting audio intervall in seconds
        start=audioStart_trial+verbOnset
        stop=audioStart_trial+objectOnset

        #restrict interval
        samples_restricted=samples_onlyTrials[(start<samples_onlyTrials["TIME"])&(samples_onlyTrials["TIME"]<stop)]

        valid_trial = samples_restricted[samples_restricted["BPOGV"] == 1].copy()
        valid_trial["screen_x"] = valid_trial["BPOGX"] * SCREEN_W
        valid_trial["screen_y"] = valid_trial["BPOGY"] * SCREEN_H
        #print(valid_trial.to_string())
        #now we check positioning in one of the 4 aois
        mappedToPositions=assign_aoi(valid_trial,positions,
                                    box_width = BOX_WIDTH,box_height = BOX_HEIGTH,
                                    x_col="screen_x",y_col="screen_y")
        #print(mappedToPositions.to_string())
        #count_inANY_AOI=mappedToPositions.shape[0]
        count_perAOI_df = (
    mappedToPositions.groupby("AOI")
    .size()
    .reset_index(name="count")
)
        #print(count_perAOI_df.to_string()) if not count_perAOI_df.empty else print("no gaze in any AOI")
        #print(count_perAOI_df)
        #percent_perAOI_df=count_perAOI_df/count_perAOI_df.sum()
        #print(percent_perAOI_df)
        
        #percentages_trial = (
        #mappedToPositions.groupby("AOI")
        #.size()
        #.reindex(["pos1", "pos2", "pos3", "pos4"], fill_value=0)
       # )

        #percentages_trial = percentages_trial / percentages_trial.sum()
        

        #before saving the percentages with objects we should also get the object strings so we can aggregate on that 
        t_pos=trial_info["target_position"].values[0]
        for  i,pos in enumerate(["pos1","pos2","pos3","pos4"]):
            #print(i)
            #in the further pipeline we use stimuli ids starting at 1 !!
            out_stimuliId.append(stimuliID+1)
            out_obj.append(trial_info["position"+str(i+1)].values[0])
            
            if i+1==t_pos:
                is_target.append(1)
            else:
                is_target.append(0)
            #outP = 0.0 if pd.isna(percentages_trial[pos]) else percentages_trial[pos]
            out_count_per_obj_per_part.append(count_perAOI_df[count_perAOI_df["AOI"]==pos]["count"].values[0] if not count_perAOI_df[count_perAOI_df["AOI"]==pos].empty else 0)
            
            #out_count_per_obj_per_part.append(outCount)
            partID.append(participantNumber)
            condition.append(condition_trial)


out_df_source = pd.DataFrame({
    "stimuliId": out_stimuliId,
    "partID": partID,
    "condition":condition,
    "obj": out_obj,
    "is_target":is_target,
    "count_per_obj_per_part": out_count_per_obj_per_part
    
})                               



#count non zero participoants ie poarticipants that contributed to the overall sum on any of the objects
count_intel_df= out_df_source.groupby(["stimuliId","condition","partID"]).agg({"count_per_obj_per_part":"sum"}).reset_index()
count_nonZeroParts_df=count_intel_df[count_intel_df["count_per_obj_per_part"] > 0].groupby(["stimuliId","condition"]).size().reset_index(name="countOfNonZeroParticipants")
count_nonZeroParts_df["countOfNonZeroParticipants"] = (
    count_nonZeroParts_df["countOfNonZeroParticipants"].fillna(0)
)
print(count_nonZeroParts_df.to_string())

out_df=out_df_source.groupby(["stimuliId","condition","obj"]).agg({"count_per_obj_per_part":"sum"}).reset_index()
out_df["percentages"]=out_df["count_per_obj_per_part"]/out_df.groupby(["stimuliId","condition"])["count_per_obj_per_part"].transform("sum")
out_df["count_per_obj_per_part"] = (
    out_df["count_per_obj_per_part"].fillna(0)
)
out_df = out_df.merge(count_nonZeroParts_df, on=["stimuliId","condition"], how="left")
out_df = out_df.merge(out_df_source[["stimuliId","condition","obj","is_target"]],on=["stimuliId","condition","obj"], how="left")
out_df.to_csv("output_all_withPercentages.csv", index=False)