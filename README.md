# Top 2500 songs from Billboard Top 50 Singles Rankings: 1973-2022

## Data

### Song Info
This repository has the code to collect information on the songs, i.e. 
- **Year,**
- **Rank,**
- **Song Name,**
- **Singer name, and the** 
- **YouTube URL**  
for Billboard Top 50 Singles Rankings from 1973-2022. The collected data is also shared here.

Download the repo and install the requirements using 
`pip install -r requirements.txt`

Follow the link to the [`create_top_50.py file`](https://github.com/Kaavya12/top-50-audios/blob/main/data/create_top_50.py) and run the code. This will generate the CSV file, that can also be found at the [`top_50s_chart.csv file`](https://github.com/Kaavya12/top-50-audios/blob/main/data/top_50s_chart.csv). To load the csv file, make sure to specific the index_col argument as `df = pd.read_csv("top_50s_chart.csv", index_col=[0,1])`

### Features Info

The features file contains the following features, for the top 2493 songs of the last 50 years (7 songs excluded): 
- **chroma_stft**
- **chroma_cens**
- **mfcc**
- **rmse**
- **zcr**
- **spectral_centroid**
- **spectral_bandwidth**
- **spectral_contrast**
- **spectral_rolloff**  
for the songs colelcted above. The data is also shared here.

Running the entire file will also save all the audio files to your local device. The raw audio hasn't been shared as it contains a massive amount of data. However, the audio files have been analysed using librosa and the extracted features are shared at [`top_50_song_features.csv`](https://github.com/Kaavya12/top-50-audios/blob/main/data/top_50_song_features.csv). The code to extract these features is shared at [`get_features.py`](https://github.com/Kaavya12/top-50-audios/blob/main/data/get_features.py).

Both of these files used in conjunction can provide immense oppotunities for exploration into trend analysis and genre classification. 

---

## Model

The [`model.py file`](https://github.com/Kaavya12/top-50-audios/blob/main/model/model.py) has the complete code for the model that has been implemented, and can be run after the requirements have been installed. 

To use the model, the fma_metadata folder is required from [fma_metadata.zip](https://os.unil.cloud.switch.ch/fma/fma_metadata.zip). This is obtained from the [mdeff/fma repo](https://github.com/mdeff/fma). This repository is a brilliant source of information, and I recommedn you check this out if you're interested in music analysis.

The `model.h5` file, `model_checkpoint`, `pipe.joblib` and `enc.joblib` are from the latest run of the model. 

---

## Predicted Data

This [`csv file`](https://github.com/Kaavya12/genre-classification-model-dash/blob/96f6e64bcb48440185ac5f324b58980d42438e50/data/top_50_predicted_data_mod10_v2.csv) in a separate repository has the complete predicted genres for the top 50 songs 

The top 3 genres have been predicted for each audio file, and can be used as needed!

---

Check out this series tolearn more about how the code is structured and how it works:
[A Genre Classification App using Tensorflow, Dash and AWS](https://medium.com/@kaavyamaha12/list/a-genre-classification-app-using-tensorflow-dash-and-aws-5c134b5ff913)
