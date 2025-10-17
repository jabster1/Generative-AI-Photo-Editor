How to Run
chmod +x run.sh               # one-time: make it executable

#Run the below script line to get output on a single image
#if you do not want all the mask images used for debugging then take out "--save-debug"
./run.sh --image images/bike_remove.jpg --prompt "remove the man on the bike in the center continue the white van body panel and road" --save-debug

# Single image
./run.sh --image images/bike_remove.jpg --prompt "remove the man on the bike in the center" --save-debug

# Whole folder
./run.sh --image images --prompt "remove people" --save-debug

# Force CPU
./run.sh --image images/photo.jpg --prompt "remove the truck" --cpu