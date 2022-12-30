# Extended Cleaned tag and Artist-Level Stratified split (eCALS)

- [Zenodo-Link](https://zenodo.org/record/7107130)

We introduce the extended tag version of CALS split (cleaned and artist-level stratified) for the Million Song Dataset (MSD). Different from the previously CALS dataset split, we provide 1054 vocabulary and caption level tag sequences instead of 50 small vocabs. Since we inherit the existing cals split, there is no difference in the test dataset. However, we use all tag annotations in the existing tag annotations `msd50, msd500, and allmusic`. This is the dataset repository for the paper: [Toward Universal Text-to-Music Retrieval](https://arxiv.org/abs/2211.14558)

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

### Dataset Loader For Classification
If you want to use this dataset for audio-language representation learning or captioning, please refer to the this [repository](https://github.com/SeungHeonDoh/music-text-representation/blob/main/mtr/contrastive/dataset.py).

```python
class ECALS_Dataset(Dataset):
    """
            data_path (str): location of msu-benchmark
            split (str): one of {TRAIN, VALID, TEST}
            sr (int): sampling rate of waveform - 16000
            num_chunks (int): chunk size of inference audio
        """
    def __init__(self, data_path, split, sr, duration, num_chunks):
        self.data_path = data_path
        self.split = split
        self.sr = sr 
        self.input_length = int(sr * duration)
        self.num_chunks = num_chunks
        self.msd_to_id = pickle.load(open(os.path.join(data_path, "lastfm_annotation", "MSD_id_to_7D_id.pkl"), 'rb'))
        self.id_to_path = pickle.load(open(os.path.join(data_path, "lastfm_annotation", "7D_id_to_path.pkl"), 'rb'))
        self.get_split()
        self.get_file_list()
    
    def get_split(self):
        track_split = json.load(open(os.path.join(self.data_path, "ecals_annotation", "ecals_track_split.json"), "r"))
        self.train_track = track_split['train_track'] + track_split['extra_track']
        self.valid_track = track_split['valid_track']
        self.test_track = track_split['test_track']
    
    def get_file_list(self):
        annotation = json.load(open(os.path.join(self.data_path, "ecals_annotation", "annotation.json"), 'r'))
        self.list_of_label = json.load(open(os.path.join(self.data_path, "ecals_annotation", "ecals_tags.json"), 'r'))
        self.tag_to_idx = {i:idx for idx, i in enumerate(self.list_of_label)}
        if self.split == "TRAIN":
            self.fl = [annotation[i] for i in self.train_track]
        elif self.split == "VALID":
            self.fl = [annotation[i] for i in self.valid_track]
        elif self.split == "TEST":
            self.fl = [annotation[i] for i in self.test_track]
        else:
            raise ValueError(f"Unexpected split name: {self.split}")
        del annotation
    
    def audio_load(self, msd_id):
        audio_path = self.id_to_path[self.msd_to_id[msd_id]]
        audio = np.load(os.path.join(self.data_path, "npy", audio_path.replace(".mp3",".npy")), mmap_mode='r')
        random_idx = random.randint(0, audio.shape[-1]-self.input_length)
        audio = torch.from_numpy(np.array(audio[random_idx:random_idx+self.input_length]))
        return audio

    def tag_to_binary(self, tag_list):
        bainry = np.zeros([len(self.list_of_label),], dtype=np.float32)
        for tag in tag_list:
            bainry[self.tag_to_idx[tag]] = 1.0
        return bainry

    def __getitem__(self, index):
        item = self.fl[index]
        tag_list = item['tag']
        binary = self.tag_to_binary(tag_list)
        audio_tensor = self.audio_load(item['track_id'])
        return {
            "audio":audio_tensor, 
            "binary":binary, 
            "tag_list":tag_list
            }

    def __len__(self):
        return len(self.fl)
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
cd dataset
wget http://millionsongdataset.com/sites/default/files/AdditionalFiles/track_metadata.db
```


```
└── dataset
    track_metadata.db
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

### Citation
Please consider citing our paper in your publications if the project helps your research. BibTeX reference is as follow.
```
@inproceedings{toward2023doh,
  title={Toward Universal Text-to-Music Retrieval},
  author={SeungHeon Doh, Minz Won, Keunwoo Choi, Juhan Nam},
  booktitle = {},
  year={2023}
}
```
