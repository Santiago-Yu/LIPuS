import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

filerkk = open("../prog_generator/records_leastSapce.txt", "r")
lines = filerkk.readlines()
records = []
chaser = 1
first = True
maxer = -1
for ler in lines:
    if first:
        first = False
        maxer = float(ler)
        continue
    ler_t = ler.split(' ')
    leastSpace = float(ler[0])
    bnumber = int(ler[1])/10
    while chaser < bnumber:
        record = {'bnumber':bnumber, 'lS':maxer}
        records.append(record)
        chaser +=1
    record = {'bnumber':bnumber, 'lS':leastSpace}
    records.append(record)


ax = sns.lineplot(x="bnumber", y="lS",data=records)
plt.show()