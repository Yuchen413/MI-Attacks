# MI-Attacks

2_dimNN: 2 dimension NN attack, using [true_confidence,predicted_confidence]
3_dimNN: 3 dimension NN attack, using [true_confidence,predicted_confidence, secondLarge_confidence]
baselineNN: Traditional NN attack using all dimensions
global_label: True label only attack
global_loss: True label's confidence only attack
instance_distance: Comparing the distance of two sample. Can be 2_dim or 3_dim(select in the code), can choose different distance calculation method(select in the code)
domain_distance: Comparing the distance of two domains. Can be 2_dim or 3_dim or all_dim(select in the code), using MMD distance.

By the way, nonMemberGenerator.py and outputCSV.py is used to get the output csv file of confidence score from target/shadow model. 
Then you may use these csv files as an input of all of attacks.
