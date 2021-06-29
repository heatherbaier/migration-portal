# Multiscale Recurrent Visual Attention to S -> R migration prediction


### Process
1. Train multiscale RAM model
2. Load the trained weights of the model in step 1
3. Run the extract_locations file to extract for every image the locations of the X glimpses 
4. Run the extract_features file to extract for every image the output of the miniConv in the Mutlscale RAM model
*I THINK THERE NEEDS TO BE A STEP IN HERE THAT SOMEHOW CONSTRUCTS THE GRAPH BASED ON THE ACTUAL LOCATIONS OF THE GLIMPSES*
5. Run the train_sr_graphsage file to train the graph network to rpedict the number of migrants beteween 2 municipalities using the image features from extract_features



### run.sh example
python3 train_munis.py (status: Done) <br>
python3 extract_locations.py (status: Done) <br>
python3 extract_features.py (status: not started) <br>
python3 train_sr_graphsage.py (status: not started) <br>