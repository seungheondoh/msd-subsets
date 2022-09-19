DATASET="./dataset"
INT_RANDOM_SEED = 42
# MUSIC_SAMPLE_RATE = 22050
MUSIC_SAMPLE_RATE = 16000
STR_CH_FIRST = 'channels_first'
STR_CH_LAST = 'channels_last'
DATA_LENGTH = MUSIC_SAMPLE_RATE * 30
INPUT_LENGTH = MUSIC_SAMPLE_RATE * 10
CHUNK_SIZE = 16
# METADATA = ['title','artist_name','release','year']
METADATA = ['artist_name','year']
BLACK_LIST = ['d-i-v-o-r-c-e', 'n metal']
MIDFEATURE =  ['key','tempo']
CONTEXTUAL = ['theme','mood','decade']
MUSICAL = ['genre','style','instrument','vocal']
CULTUREAL = ['language','location']

TOKEN_DICT = {    
    "artist_name":'[ARTIST_NAME]', 
    "release":'[RELEASE]', 
    "title":'[TITLE]', 
    "year":'[YEAR]',
    "theme":'[THEME]',
    "mood":'[MOOD]',  
    "genre":'[GENRE]', 
    "style":'[STYLE]', 
    "instrument":'[INSTRUMENT]', 
    "decade":'[DECADE]', 
    "language":'[LANGUAGE]',
    "location":'[LOCATION]',
    "vocal":'[VOCAL]',
    "tempo":'[TEMPO]',
    "key":'[KEY]'
}

LASTFM_TAG_INFO = {
    '00s': "decade",
    '60s': "decade",
    '70s': "decade",
    '80s': "decade",
    '90s': "decade",
    'acoustic': "instrument",
    'alternative': "genre",
    'alternative rock': "genre",
    'ambient': "genre",
    'beautiful': "mood",
    'blues': "genre",
    'catchy': "mood",
    'chill': "mood",
    'chillout': "mood",
    'classic rock': "genre",
    'country': "genre",
    'dance': "mood",
    'easy listening': "mood",
    'electro': "genre",
    'electronic': "genre",
    'electronica': "genre",
    'experimental': "genre",
    'female vocalist': "vocal",
    'female vocalists': "vocal",
    'folk': "genre",
    'funk': "genre",
    'guitar': "instrument",
    'happy': "mood",
    'hard rock': "genre",
    'heavy metal': "genre",
    'hip-hop': "genre",
    'house': "genre",
    'indie': "genre",
    'indie pop': "genre",
    'indie rock': "genre",
    'instrumental': "genre",
    'jazz': "genre",
    'male vocalists': "vocal",
    'mellow': "mood",
    'metal': "genre",
    'oldies': "mood",
    'party': "mood",
    'pop': "genre",
    'progressive rock': "genre",
    'punk': "genre",
    'rnb': "genre",
    'rock': "genre",
    'sad': "mood",
    'sexy': "mood",
    'soul': "genre"
}