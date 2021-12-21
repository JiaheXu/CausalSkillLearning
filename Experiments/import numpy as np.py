import numpy as np 
roboturk_labels = {}
roboturk_labels['0'] = 'Place to Left'
roboturk_labels['1'] = 'Place to Right'
roboturk_labels['2'] = 'Noisy move'
roboturk_labels['3'] = 'Place to Left'
roboturk_labels['4'] = 'Place to Left'
roboturk_labels['5'] = 'Noisy move'
roboturk_labels['6'] = 'Noisy move'
roboturk_labels['7'] = 'Place to Left'
roboturk_labels['8'] = 'Noisy move'
roboturk_labels['9'] = 'Place to Left'
roboturk_labels['10'] = 'Place to Left'
roboturk_labels['11'] = 'Reach from Left'
roboturk_labels['12'] = 'Reach from Left'
roboturk_labels['13'] = 'Noisy move'
roboturk_labels['14'] = 'Noisy move'
# roboturk_labels['15'] = 'Pick and place'
roboturk_labels['16'] = 'Noisy move'
# roboturk_labels['17'] = 'Move to robot left nad then back right'
roboturk_labels['18'] = 'Grasp and Lift'
roboturk_labels['19'] = 'Reach from Left'
roboturk_labels['20'] = 'Place to Left'
roboturk_labels['21'] = 'Place to Left'
roboturk_labels['22'] = 'Place to Left'
roboturk_labels['23'] = 'Noisy move'
roboturk_labels['24'] = 'Noisy move'
roboturk_labels['25'] = 'Reach from Left'
roboturk_labels['26'] = 'Place to Left'
roboturk_labels['27'] = 'Place to Left'
roboturk_labels['28'] = 'Place to Left'
roboturk_labels['29'] = 'Place to Left'
roboturk_labels['30'] = 'Place to Left'
roboturk_labels['31'] = 'Noisy move'
# roboturk_labels['32'] = 'Move to robot left then back to mid'
roboturk_labels['33'] = 'Place to Right'
roboturk_labels['34'] = 'Noisy move'
roboturk_labels['35'] = 'Place to Left'
roboturk_labels['36'] = 'Place to Right'
roboturk_labels['37'] = 'Grasp and Lift'
roboturk_labels['38'] = 'Place to Right'
roboturk_labels['39'] = 'Place to Left'
roboturk_labels['40'] = 'Reach from Left'
roboturk_labels['41'] = 'Grasp and Lift'
roboturk_labels['42'] = 'Place to Left'
roboturk_labels['43'] = 'Noisy move'
roboturk_labels['44'] = 'Place to Left'
roboturk_labels['45'] = 'Grasp and Lift'
roboturk_labels['46'] = 'Grasp and Lift'
roboturk_labels['47'] = 'Place to Left'
roboturk_labels['48'] = 'Place to Left'
roboturk_labels['49'] = 'Grasp and Lift'


roboturk_inverse_labels = {}
for k,v in roboturk_labels.items():
    print(k,v)
    if v not in roboturk_inverse_labels.keys():
        roboturk_inverse_labels[v] = [k]
    else:
        roboturk_inverse_labels[v].append(k)
