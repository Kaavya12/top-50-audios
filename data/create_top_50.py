from __future__ import unicode_literals

import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import yt_dlp as youtube_dl

years = list(range(1973,2023))

base_url = "https://en.wikipedia.org/wiki/Billboard_Year-End_Hot_100_singles_of_"
songs = {}
singers = {}

for year in years:
  names = []
  singer_names = []
  r = requests.get(f"{base_url}{year}")
  soup = BeautifulSoup(r.content)
  table = soup.findAll('table', {'class': 'wikitable'})[0]
  for row in table.findAll('tr'):
    links = row.findAll('a')
    if links:
      if len(links) > 2:
        song = links[0].contents[0]
        singer = ""
        for link in links[1:]:
          singer += f"{link.contents[0]},"
      elif len(links) == 2:
        song, singer = links[0].contents[0], links[1].contents[0]
      else:
        song, singer = links[0].contents[0], None
      names.append(song)
      singer_names.append(singer)
  songs[year] = names
  singers[year] = singer_names
  
"""
for year, hits in songs.items():
  print(f"{year}: {len(hits)} - Top: {hits[0]}")
"""

songs_df = pd.DataFrame(songs).transpose()
songs_df.columns = [i for i in range(1,101)]
singers_df = pd.DataFrame(singers).transpose()
singers_df.columns = [i for i in range(1,101)]

songs_df = songs_df.stack()
singers_df = singers_df.stack()
song_details = pd.concat([songs_df, singers_df], axis=1, keys=["song", "singer"])

def fetch_yt_urls(item):
  song = item["song"]
  singer = item["singer"]
  year = item.name[0]
  song_search = "+".join(song.split())
  r = requests.get(f"https://www.google.com/search?q={song_search}+{year}+{singer}+song")
  d = {
      "%20":" ", "%21":"!", "%22":'"', "%23":"#", "%24":"$", "%25":"%", "%26":"&", "%27":"'", "%28":"(", "%29":")", "%2A":"*", "%2B":"+", "%2C":",", "%2D":"-", "%2E":".", "%2F":"/", "%3A":":", "%3B":";", "%3C":"<", "%3D":"=", "%3E":">", "%3F":"?"
    }
  soup = BeautifulSoup(r.content)
  for link in soup.findAll('a'):
    if link['href'].startswith("/url?q=https://www.youtube.com/watch"):
      src_link = link['href'][7:]
      for key, value in d.items():
        src_link = src_link.replace(key, value)
      src_link = src_link.partition("&")[0]
      return src_link
      
  return None

yt_urls = song_details.apply(fetch_yt_urls, axis=1)
all_data = pd.concat([song_details, yt_urls], axis=1)
all_data.columns = ["song", "singer", "urls"]

top_50 = all_data[np.in1d(all_data.index.get_level_values(1), range(1,51))]

#the below is to fill up the few null values that I encountered
top_50.loc[(1994, 24), "urls"] = "https://www.youtube.com/watch?v=a02dBbBGSPg"
top_50.loc[(1995, 41), "urls"] = "https://www.youtube.com/watch?v=6exsatE-DUk"
top_50.loc[(1998, 9), "urls"] =  "https://www.youtube.com/watch?v=DIpQ4AZSAf8"
top_50.loc[(2013, 20), "urls"] = "https://www.youtube.com/watch?v=IsUsVbTj2AY"
top_50.loc[(2018, 45), "urls"] = "https://www.youtube.com/watch?v=SA7AIQw-7Ms" 

top_50.to_csv("top_50s_chart.csv")

def save_audios(row):
  year = row.name[0]
  rank = row.name[1]
  ydl_opts = {
      'format': 'bestaudio/best',
      'postprocessors': [{
          'key': 'FFmpegExtractAudio',
          'preferredcodec': 'mp3',
          'preferredquality': '192',
      }],
      'outtmpl': f'audios/{year}/{rank}'
    }
  
  try:
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
      ydl.download([row["urls"]])
  except:
    pass

top_50.apply(save_audios, axis=1)
