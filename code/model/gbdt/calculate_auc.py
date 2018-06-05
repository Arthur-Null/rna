file = open("result", 'r')

auc = []
all = 0
for line in file.readlines():
    pos = line.find(':')
    num = float(line[pos + 1: pos + 7])
    all += num

print(all, all / 37)