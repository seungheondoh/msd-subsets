import os
import pandas as pd
from msd_preprocessor import MSD_processor
from constants import DATASET

def main():
    MSD_processor(msd_path= os.path.join(DATASET))

if __name__ == '__main__':
    main()