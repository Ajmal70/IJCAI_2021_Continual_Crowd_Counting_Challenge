
#Downloading contiguous videos in zip
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1phQi86FvLXBoOeh9jeZ_-MomUbcGW04K' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1phQi86FvLXBoOeh9jeZ_-MomUbcGW04K" -O data.zip
#Unzipping the videos
unzip data.zip
#Removing the zip file
rm data.zip
#
