# CycleGAN-project
New dataset: Real2Anime
```
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=18QG9SCz2460DNXfPnq_2moupwd3kjxKr' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=18QG9SCz2460DNXfPnq_2moupwd3kjxKr" -O Real2Anime.zip && rm -rf /tmp/cookies.txt
```

    .
    ├── datasets                   
    |   ├── Real2Anime
    |   |   ├── train              # Training
    |   |   |   ├── A              # 6656 real photos, 256*256
    |   |   |   └── B              # 1650 Shinkai smooth photo, 256*256
    |   |   └── test               # Testing
    |   |   |   ├── A              # around 150 real scenes that seem to be easier to transform, mostly compressed to 256*256 
    |   |   |   └── B              # around 200 cartoon phones 
