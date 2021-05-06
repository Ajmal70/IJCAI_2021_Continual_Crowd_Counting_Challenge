
#Downloading contiguous videos in zip
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1N71uZvw9Z3wqHRucJKmvmOwDWNIfK_1y' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1N71uZvw9Z3wqHRucJKmvmOwDWNIfK_1y" -O data.zip
#Unzipping the videos
unzip data.zip
#Removing the zip file
rm data.zip
#
