file = open("result", 'r')

auc = []
all = 0
for line in file.readlines():
    pos = line.find('c')
    if pos == -1:
        continue
    num = float(line[pos + 2: ])
    auc.append(num)
    if len(auc) == 37:
        print(sum(auc) / 37)
        auc = []

