import os
import pickle
import random
import sqlite3
import torch
import json
import pandas as pd
import numpy as np
import multiprocessing
from collections import Counter
from functools import partial
from contextlib import contextmanager
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from audio_utils import load_audio
from sklearn.preprocessing import MultiLabelBinarizer

from constants import DATASET, DATA_LENGTH, STR_CH_FIRST, MUSIC_SAMPLE_RATE, BLACK_LIST, LASTFM_TAG_INFO

NaN_to_emptylist = lambda d: d if isinstance(d, list) or isinstance(d, str) else []
flatten_list_of_list = lambda l: [item for sublist in l for item in sublist]

def tag_normalize(tag):
    tag = tag.replace("'n'","").replace("'","").replace("(","").replace(")","").replace("/"," ").replace("-"," ").replace(" & ","n").replace("&", "n")
    tag = unique_word(tag)
    return tag

def unique_word(tag):
    unique_tag, remove_dix = [], None
    token = tag.split()
    for idx, i in enumerate(token):
        if len(i) == 1:
            unique_tag.append(token[idx] + token[idx+1])
            remove_dix = idx + 1
        else:
            unique_tag.append(i)
    if remove_dix:
        unique_tag.remove(token[remove_dix])
    return " ".join(unique_tag)

def _remove(tag_list):
    return [i for i in tag_list if i not in BLACK_LIST]

def getMsdInfo(msd_path):
    con = sqlite3.connect(msd_path)
    msd_db = pd.read_sql_query("SELECT * FROM songs", con)
    msd_db = msd_db.set_index('track_id')
    return msd_db

def _json_dump(path, item):
    with open(path, mode="w") as io:
        json.dump(item, io, indent=4)

@contextmanager
def poolcontext(*args, **kwargs):
    pool = multiprocessing.Pool(*args, **kwargs)
    yield pool
    pool.terminate()
    
def msd_resampler(_id, path):
    save_name = os.path.join(DATASET,'npy', path.replace(".mp3",".npy"))
    try:
        src, _ = load_audio(
            path=os.path.join(DATASET,'songs',path),
            ch_format= STR_CH_FIRST,
            sample_rate= MUSIC_SAMPLE_RATE,
            downmix_to_mono= True)
        if src.shape[-1] < DATA_LENGTH: # short case
            pad = np.zeros(DATA_LENGTH)
            pad[:src.shape[-1]] = src
            src = pad
        elif src.shape[-1] > DATA_LENGTH: # too long case
            src = src[:DATA_LENGTH]
        
        if not os.path.exists(os.path.dirname(save_name)):
            os.makedirs(os.path.dirname(save_name))
        np.save(save_name, src.astype(np.float32))
    except:
        os.makedirs(os.path.join(DATASET,"error"), exist_ok=True)
        np.save(os.path.join(DATASET,"error", _id + ".npy"), _id) # check black case

def binary_df_to_list(binary, tags, indices, data_type):
    list_of_tag = []
    for bool_tags in binary:
        list_of_tag.append([tags[idx] for idx, i in enumerate(bool_tags) if i] )
    df_tag_list = pd.DataFrame(index=indices, columns=[data_type])
    df_tag_list.index.name = "track_id"
    df_tag_list[data_type] = list_of_tag
    df_tag_list['is_'+ data_type] = [True for i in range(len(df_tag_list))]
    return df_tag_list

def lastfm_processor(lastfm_path):
    """
    input: lastfm_path 
    return: pandas.DataFrame => index: msd trackid, columns: list of tag
            TRAAAAK128F9318786	[rock, alternative rock, hard rock]
            TRAAAAW128F429D538	[hip-hop]
    """
    lastfm_tags = open(os.path.join(lastfm_path, "50tagList.txt"),'r').read().splitlines()
    lastfm_tags = [i.lower() for i in lastfm_tags]
    # lastfm split and 
    train_list = pickle.load(open(os.path.join(lastfm_path, "filtered_list_train.cP"), 'rb'))
    test_list = pickle.load(open(os.path.join(lastfm_path, "filtered_list_test.cP"), 'rb'))
    msd_id_to_tag_vector = pickle.load(open(os.path.join(lastfm_path, "msd_id_to_tag_vector.cP"), 'rb'))
    total_list = train_list + test_list
    binary = [msd_id_to_tag_vector[msdid].astype(np.int16).squeeze(-1) for msdid in total_list]
    track_split = {
        "train_track": train_list[0:201680],
        "valid_track": train_list[201680:],
        "test_track": test_list,
    }
    
    _json_dump(os.path.join(lastfm_path, "lastfm_tags.json"), lastfm_tags)
    _json_dump(os.path.join(lastfm_path, "lastfm_tag_info.json"), LASTFM_TAG_INFO)
    _json_dump(os.path.join(lastfm_path, "lastfm_track_split.json"), track_split)

    lastfm_binary = pd.DataFrame(binary, index=total_list, columns=lastfm_tags)
    df_lastfm = binary_df_to_list(binary=binary, tags=lastfm_tags, indices=total_list, data_type="lastfm")
    return df_lastfm, track_split

def cals_processor(cals_path):
    train_ids = np.load(os.path.join(cals_path, "train_ids.npy"))
    train_binary = np.load(os.path.join(cals_path, "train_binaries.npy"))
    valid_ids = np.load(os.path.join(cals_path, "valid_ids.npy"))
    valid_binary = np.load(os.path.join(cals_path, "valid_binaries.npy"))
    test_ids = np.load(os.path.join(cals_path, "test_ids.npy"))
    test_binary = np.load(os.path.join(cals_path, "test_binaries.npy"))
    cals_ids = list(train_ids) + list(valid_ids) + list(test_ids)
    cals_binary = np.vstack([train_binary,valid_binary,test_binary])
    cals_tags = list(np.load(os.path.join(cals_path, "cals_tags.npy")))
    ids_to_tag = {}
    for ids, binary in zip(cals_ids, cals_binary):
        ids_to_tag[ids] = {
            "cals":[cals_tags[idx] for idx, i in enumerate(binary) if i],
            "is_cals":True
        }
    df_cals = pd.DataFrame(ids_to_tag).T
    df_cals.index.name = "track_id"
    return df_cals

def allmusic_processor(allmusic_path):
    """
    input: allmusic_path 
    return: pandas.DataFrame => index: msd trackid, columns: list of tag
            TRWYIGP128F1454835	[Pop/Rock, Electronic, Adult Alternative Pop/R...
            TRGFXIU128F1454832	[Pop/Rock, Electronic, Adult Alternative Pop/R...
    """
    df_all = pd.read_hdf(os.path.join(allmusic_path, 'ground_truth_assignments/AMG_Multilabel_tagsets/msd_amglabels_all.h5'))
    tag_stats, tag_dict = {}, {}
    for category in df_all.columns:
        df_all[category] = df_all[category].apply(NaN_to_emptylist)
        df_all[category] = df_all[category].map(lambda x: list(map(str.lower, x)))
        tag_stats[category[:-1]] = {i:j for i,j in Counter(flatten_list_of_list(df_all[category])).most_common()}
        for tag in set(flatten_list_of_list(df_all[category])):
            tag_dict[tag] = category[:-1]
    _json_dump(os.path.join(allmusic_path, "allmusic_tags.json"), list(tag_dict.keys()))
    _json_dump(os.path.join(allmusic_path, "allmusic_tag_info.json"), tag_dict)
    _json_dump(os.path.join(allmusic_path, "allmusic_tag_stats.json"), tag_stats)

    tag_list = df_all['genres']+df_all['styles']+df_all['moods']+df_all['themes']
    df_allmusic = pd.DataFrame(index=df_all.index, columns=["allmusic"])
    df_allmusic["allmusic"] = tag_list
    df_allmusic['is_allmusic'] = [True for i in range(len(df_allmusic))]
    return df_allmusic

def msd500_processor(msd500_path):
    msd500_tags = pd.read_csv(os.path.join(msd500_path,"selected_tags.tsv"), sep='\t', header=None)
    msd500_map = {'mood':'mood', 'instrument':'instrument', 'activity':'theme', 
            'language':'language', 'location':'location', 'decade':'decade', 'genre':'genre'}
    msd500_tag_info = {i:msd500_map[j.split("/")[0]] for i,j in zip(msd500_tags[0], msd500_tags[1])}
    msd500_anno = pd.read_csv(os.path.join(msd500_path,"track_tags.tsv"), sep="\t", header=None)
    use_tag = list(msd500_tag_info.keys())
    msd500_anno = msd500_anno.set_index(2)
    msd500_anno = msd500_anno.loc[use_tag]
    item_dict = {i:[] for i in msd500_anno[0]}
    for _id, tag in zip(msd500_anno[0], msd500_anno.index):
        item = item_dict[_id].copy()
        item.append(tag)
        item_dict[_id] = list(set(item))

    df_msd500 = pd.DataFrame(index=item_dict.keys())
    df_msd500['msd500'] = item_dict.values()
    df_msd500['is_msd500'] = [True for i in range(len(df_msd500))]
    df_msd500.index.name = "track_id"
    msd500_tag_stat = {i:j for i,j in Counter(flatten_list_of_list(df_msd500['msd500'])).most_common()}
    _json_dump(os.path.join(msd500_path, "msd500_tag_info.json"), msd500_tag_info)
    _json_dump(os.path.join(msd500_path, "msd500_tags.json"), list(msd500_tags[0]))
    _json_dump(os.path.join(msd500_path, "msd500_tag_stats.json"), msd500_tag_stat)
    return df_msd500

def _check_mp3_file(df_msd, id_to_path, MSD_id_to_7D_id):
    mp3_path, error_id = {}, []
    for msdid in df_msd.index:
        try:
            mp3_path[msdid] = id_to_path[MSD_id_to_7D_id[msdid]]
        except:
            error_id.append(msdid)
    df_msd = df_msd.drop(error_id)
    return df_msd, mp3_path

def _track_split(df_target, msd_path, types = "ecals"):
    track_split = {}
    if types == "ecals":
        df_target = df_target[df_target['tag'].apply(lambda x: len(x) != 0)]
    for i in set(df_target['splits']):
        track_list = list(df_target[df_target['splits'] == i].index)
        if i == "TRAIN":
            track_split['train_track'] = track_list
        elif i == "VALID":
            track_split['valid_track'] = track_list
        elif i == "TEST":
            track_split['test_track'] = track_list
        elif i == "STUDENT":
            track_split['extra_track'] = track_list
    _tag_stat = {i:j for i,j in Counter(flatten_list_of_list(list(df_target['tag']))).most_common()}
    track_list = track_split['train_track'] + track_split['valid_track']+ track_split['test_track']
    print("finish msd extraction", len(track_list), "extra_track: ", len(track_split['extra_track']), "tag: ", len(_tag_stat))
    
    _json_dump(os.path.join(msd_path, f"{types}_track_split.json"), track_split)
    _json_dump(os.path.join(msd_path, f"{types}_tags.json"), list(_tag_stat.keys()))
    _json_dump(os.path.join(msd_path, f"{types}_tag_stats.json"), _tag_stat)
    return track_split

def _check_stat(df, track_list):
    df_test = df.loc[track_list]
    save_tag = set(df_test.T.loc[df_test.sum() > 2].index)
    return save_tag

def filtering(df_tags, tr_track, va_track, te_track):
    merge_tag = df_tags['cals'] + df_tags['lastfm'] + df_tags['msd500'] + df_tags['allmusic']
    merge_tag = merge_tag.apply(lambda x: _remove(x))
    merge_tag = merge_tag.apply(lambda x: list(map(tag_normalize, x)))
    tag_list = merge_tag.apply(set).apply(list)
    mlb = MultiLabelBinarizer()
    binary = mlb.fit_transform(tag_list)
    df = pd.DataFrame(binary, index=list(merge_tag.index), columns=mlb.classes_)
    tr_save = _check_stat(df, tr_track)
    va_save = _check_stat(df, va_track)
    te_save = _check_stat(df, te_track)
    tag = list(tr_save & va_save & te_save)
    df_all = df[tag]
    df_binary = df_all.loc[df_all.sum(axis=1) > 0]
    filtered_tag = []
    for idx in range(len(df_all)):
        item = df_all.iloc[idx]
        filtered_tag.append(list(item[item == 1].index))
    return filtered_tag, df_binary


def MSD_processor(msd_path):
    meta_path = os.path.join(msd_path, "track_metadata.db")
    lastfm_path = os.path.join(msd_path, "lastfm_annotation")
    allmusic_path = os.path.join(msd_path, "allmusic_annotation")
    msd500_path = os.path.join(msd_path, "msd500_annotation")
    cals_path = os.path.join(msd_path, "cals_annotation")
    ecals_path = os.path.join(msd_path, "ecals_annotation")
    os.makedirs(ecals_path, exist_ok=True)

    MSD_id_to_7D_id = pickle.load(open(os.path.join(lastfm_path, "MSD_id_to_7D_id.pkl"), 'rb'))
    id_to_path = pickle.load(open(os.path.join(lastfm_path, "7D_id_to_path.pkl"), 'rb'))
    lastfm_tags = [i.lower() for i in open(os.path.join(lastfm_path, "50tagList.txt"),'r').read().splitlines()]
    cals_split = pd.read_csv(os.path.join(cals_path, "msd_splits.tsv"), sep="\t").rename(columns={"clip_ids":"track_id"}).set_index("track_id")
    df_msdmeta = getMsdInfo(meta_path)
    df_cals = cals_processor(cals_path)
    df_lastfm, _ = lastfm_processor(lastfm_path)
    df_msd500 = msd500_processor(msd500_path)
    df_allmusic = allmusic_processor(allmusic_path)
    cals_lastfm = pd.merge(df_cals, df_lastfm, how='outer',on='track_id')
    cals_lastfm_msd500 = pd.merge(cals_lastfm, df_msd500, how='outer',on='track_id')
    cals_lastfm_msd500_allmusic = pd.merge(cals_lastfm_msd500, df_allmusic, how='outer',on='track_id')
    df_tags = pd.merge(cals_split, cals_lastfm_msd500_allmusic, how='outer',on='track_id')
    df_tags['length'] = df_tags['length'] / 22050

    for column in ["cals","lastfm","msd500","allmusic","is_cals","is_lastfm","is_msd500","is_allmusic"]:
        if "is_" in column:
            df_tags[column] = df_tags[column].fillna(False)
        else:
            df_tags[column] = df_tags[column].apply(NaN_to_emptylist)
    df_merge = pd.merge(df_tags, df_msdmeta, how='left',on='track_id')
    df_merge['splits'] = df_merge['splits'].fillna("NONE")
    df_final = df_merge[df_merge['splits'] != "NONE"]
    
    target_col = ["splits","length",'cals',"lastfm","msd500","allmusic","is_cals","is_lastfm","is_msd500","is_allmusic","release","artist_name","year","title"]
    df_target = df_final[target_col]
    df_target, mp3_path = _check_mp3_file(df_target, id_to_path, MSD_id_to_7D_id)

    with poolcontext(processes=multiprocessing.cpu_count()) as pool:
        pool.starmap(msd_resampler, zip(list(mp3_path.keys()),list(mp3_path.values())))
    print("finish extract")
    
    error_ids = [msdid.replace(".npy","") for msdid in os.listdir(os.path.join(msd_path, 'error'))]
    df_target = df_target.drop(error_ids) # drop errors   
    tr_track = list(df_target[df_target['splits'] == "TRAIN"].index)
    va_track = list(df_target[df_target['splits'] == "VALID"].index)
    te_track = list(df_target[df_target['splits'] == "TEST"].index)

    filtered_tag, df_binary = filtering(df_target, tr_track, va_track, te_track)
    df_target['tag'] = filtered_tag
    binary_error = [i for i in error_ids if i in df_binary.index]
    df_binary = df_binary.drop(binary_error) # drop errors   

    df_binary.to_csv(os.path.join(ecals_path, 'ecals_binary.csv'))
    df_target['track_id'] = df_target.index
    track_split = _track_split(df_target, ecals_path, types = "ecals")
    ecals_track = track_split['train_track'] + track_split['valid_track'] + track_split['test_track'] + track_split['extra_track']
    annotation_dict = df_target[["tag","release","artist_name","year","title",'track_id']].to_dict('index') # for small
    target_anotation_dict = {i:annotation_dict[i] for i in ecals_track}
    print(len(target_anotation_dict))
    with open(os.path.join(ecals_path, f"annotation.json"), mode="w") as io:
        json.dump(target_anotation_dict, io)
        
    # torch.save(annoation_dict, os.path.join(msd_path, 'annotation.pt')) # 183M