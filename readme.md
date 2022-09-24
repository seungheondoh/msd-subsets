# Extended Cleaned tag and Artist-Level Stratified split (eCALS)

We introduce the extended tag version of CALS split (cleaned and artist-level stratified) for the Million Song Dataset (MSD). Different from the previously CALS dataset split, we provide 1054 vocabulary and caption level tag sequences instead of 50 small vocabs. Since we inherit the existing cals split, there is no difference in the test dataset. However, we use all tag annotations in the existing tag annotations `msd50, msd500, and allmusic`.

<p align = "center">
    <img src = "https://i.imgur.com/jdYrysT.png">
</p>


### Example of data annotation

Dataset is `key-value` type. The key is `msdid`, and value is python dictionary (item). Item consists of tag annotation data, and artist name, track title, and year meta data.

```
{
    "TRSAGNY128F425391E": 
        {
            'tag': 
                ['aggressive', 'confrontational', 'energetic', 'alternative indie rock', 'self conscious', 'rowdy', 'bravado', 'pop rock', 'hardcore punk', 'passionate', 'confident', 'gutsy', 'swaggering', 'earnest', 'urgent', 'anguished distraught', 'straight edge', 'cathartic', 'punk', 'brash', 'rebellious', 'dramatic', 'alternative pop rock', 'street smart', 'summery', 'knotty', 'volatile', 'fiery', 'punk new wave', 'angry'], 
            'release': 'Sink With Kalifornija', 
            'artist_name': 'Youth Brigade', 
            'year': 1984, 
            'title': 'What Are You Fighting For?', 
            'track_id': 'TRSAGNY128F425391E'
        }
}
```

### Dataset stat

```
Train Track: 444865 (== CALS train + student)
Valid Track: 34481 (== CALS valid track) 
Test Track: 34631 (== CALS test track)
Unique Tag: 1054
Unique Tag Caption: 139541
Unique Artist: 32650
Unique Album : 89920
Unique Year: 90
```

### Download Source Dataset from Zenodo

```
wget https://zenodo.org/record/7107130/files/dataset.tar.gz
tar -xvf dataset.tar.gz
```


```
└── dataset
    ├── allmusic_annotation
    │   └── ground_truth_assignments
    │       ├── AMG_Multilabel_tagsets
    │       │   ├── msd_amglabels_all.h5
    ...
    ├── cals_annotation
    │   ├── cals_error.npy
    │   ├── cals_tags.npy
    ...
    ├── lastfm_annotation
    │   ├── 50tagList.txt
    │   ├── filtered_list_test.cP
    │   ├── filtered_list_train.cP
    │   ├── msd_id_to_tag_vector.cP
    ...
    ├── msd500_annotation
    │   ├── dataset_stats.txt
    │   ├── selected_tags.tsv
    │   └── track_tags.tsv
    └── ecals_annotation
        ├── annotation.json
        ├── ecals_tags.json
        ├── ecals_tag_stats.json
        ├── ecals_track_split.json
        └── multiquery_samples.json
```

### Run Preprocessing Code
```
cd preprocessing
python main.py
```

### MSD audio
Due to copyright issue, we don't provide audio data in this page.

### Reference