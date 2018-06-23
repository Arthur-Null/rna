import numpy as np
import matplotlib.pyplot as plt
deep = [0.8227351131478, 0.7142275787527184, 0.8053221601489758, 0.7583656109971899, 0.9786397069144607, 0.8316271717109706, 0.7809487737903608, 0.7906119775080522, 0.7518488002893775, 0.7571692345151297, 0.9453193913554837, 0.8116621983914208, 0.8245104175981399, 0.7818267286916072, 0.9422345564806249, 0.88373950423176, 0.9584599724047433, 0.8043074304073659, 0.8118163775360188, 0.8149479500564764, 0.7420905232142597, 0.8174930167597765, 0.8544845604380333, 0.9665980857411747, 0.8089674167472923, 0.8158415483185926, 0.9146690367360758, 0.9150527192008879, 0.9090811133398539, 0.9818274073517254, 0.9462480522935431, 0.9356910883758497, 0.8907498643675127, 0.7697354153112124, 0.9235623088098078, 0.8658823529411764, 0.7640172676485526]
gbdt = [0.7905, 0.5816, 0.8523, 0.6843, 0.9769, 0.7833, 0.658, 0.6673, 0.7418, 0.6877, 0.9062, 0.741, 0.8165, 0.7338, 0.9606, 0.8546, 0.9548, 0.6327, 0.6811, 0.6435, 0.5946, 0.722, 0.8048, 0.9515, 0.7289, 0.7947, 0.9159, 0.8901, 0.8931, 0.9725, 0.9381, 0.9263, 0.8882, 0.672, 0.9267, 0.8431, 0.6209]
svm = [0.689567430025, 0.651488616462, 0.700909090909, 0.799368088468, 0.878154917319, 0.81243063263, 0.733399405352, 0.69304099142, 0.647413793103, 0.632042253521, 0.734951456311, 0.656862745098, 0.868593955322, 0.666666666667, 0.744554455446, 0.736692015209, 0.87890625, 0.626297577855, 0.659784560144, 0.659728506787, 0.674115456238, 0.67405355494, 0.721094439541, 0.883307573416, 0.689250225836, 0.74543946932, 0.813432835821, 0.705490848586, 0.799442896936, 0.881024096386, 0.843894899536, 0.847115384615, 0.828804347826, 0.726968174204, 0.849506578947, 0.75625, 0.676390154968]

pot_list = ['AGO1', 'AGO2', 'AGO3', 'ALKBH5', 'AUF1', 'C17ORF85', 'C22ORF28', 'CAPRIN1', 'DGCR8', 'EIF4A3', 'EWSR1',
                'FMRP', 'FOX2', 'FUS', 'FXR1', 'FXR2', 'HNRNPC', 'HUR', 'IGF2BP1', 'IGF2BP2', 'IGF2BP3', 'LIN28A',
                'LIN28B',
                'METTL3', 'MOV10', 'PTB', 'PUM2', 'QKI', 'SFRS1', 'TAF15', 'TDP43', 'TIA1', 'TIAL1', 'TNRC6', 'U2AF65',
                'WTAP', 'ZC3H7B']





ind = 4 * np.arange(len(deep))  # the x locations for the groups
width = 1 # the width of the bars

fig, ax = plt.subplots(figsize=(15, 10))

rects1 = ax.barh(ind, svm, width,
                color='SkyBlue', label='Deep')
# rects2 = ax.barh(ind + width, gbdt, width,
#                 color='IndianRed', label='GBDT')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('accuracy')
ax.set_title('accuracy of SVM')
ax.set_yticks(ind)
ax.set_yticklabels(pot_list)
#ax.legend()


# def autolabel(rects, xpos='center'):
#     """
#     Attach a text label above each bar in *rects*, displaying its height.
#
#     *xpos* indicates which side to place the text w.r.t. the center of
#     the bar. It can be one of the following {'center', 'right', 'left'}.
#     """
#
#     xpos = xpos.lower()  # normalize the case of the parameter
#     ha = {'center': 'center', 'right': 'left', 'left': 'right'}
#     offset = {'center': 0.5, 'right': 0.57, 'left': 0.43}  # x_txt = x + w*off
#
#     # for rect in rects:
#     #     height = rect.get_width()
#     #     ax.text(rect.get_y() + rect.get_width()*offset[xpos], 1.01*height,
#     #             '{}'.format(height), ha=ha[xpos], va='bottom')
#
#
# autolabel(rects1, "left")
# autolabel(rects2, "right")

plt.savefig('result.jpg', bbox_inches='tight')
plt.show()