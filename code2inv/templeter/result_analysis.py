
class Record:
    def __init__(self):
        self.time = 0.0
        self.solved = False
        self.solution = None
        self.pnumber = "0"
        self.ce_I = 0
        self.candidates = 0
        self.ce_T = 0
        self.ce_F = 0
        self.z3_quries = 0


class Data:
    def __init__(self):
        self.solved = set()
        self.records = {}


def getPnumber(line1):
    # Solving 1.c.json:::
    parts = line1.split('.')[0]
    parts2 =parts.split(' ')[-1]
    return parts2

def getSTS(line2):
    # solved:False   timeUsd:600.1813535690308   solution:None
    parts = line2.split('   ')
    solved = parts[0].split(':')[-1]
    timer = parts[1].split(':')[-1]
    solution = parts[2].split(':')[-1]
    return solved, timer, solution

def getCITFZ(line3):
    # Counter({'ce-I:': 228598, 'boogie_result': 2511, 'ce-T:': 2510, 'ce-F:': 2496, 'actual_z3': 1564})
    parts = line3.split(',')
    ce_I, candidates, ce_T, ce_F, z3_quries = 0,0,0,0,0
    for part in parts:
        part = part.replace('})', '')
        if 'ce-I' in part:
            ce_I = part.split(':')[-1]
        elif 'boogie_result' in part:
            candidates = part.split(':')[-1]
        elif 'ce-T' in part:
            ce_T = part.split(':')[-1]
        elif 'ce_F' in part:
            ce_F = part.split(':')[-1]
        elif 'actual_z3' in part:
            z3_quries = part.split(':')[-1]
    return ce_I, candidates, ce_T, ce_F, z3_quries

def getRecord(line1, line2, line3):
    record = Record()

    pnumber = getPnumber(line1)
    record.pnumber = pnumber


    solved, timer, solution = getSTS(line2)

    if solved == "True":
        record.solved = True
    else:
        record.solved = False
    record.time=float(timer)
    record.solution = solution


    ce_I, candidates, ce_T, ce_F, z3_quries = getCITFZ(line3)
    record.ce_I = int(ce_I)
    record.candidates = int(candidates)
    record.ce_T = int(ce_T)
    record.ce_F = int(ce_F)
    record.z3_quries = int(z3_quries)

    return record

def parseData(pather):
    res = Data()
    filer = open(pather, "r")
    lines = filer.readlines()
    begin = False
    i = 0
    while i < len(lines):
        if "\n" == lines[i]:
            i += 1
            continue
        if "##################################################\n" == lines[i]:
            break
        if "#################################################################################\n" == lines[i]:
            begin = True
            i+=1
            continue
        if not begin:
            i+=1
            continue

        # three lines for each record.

        record = getRecord(lines[i], lines[i+1], lines[i+2])
        if record.solved:
            res.solved.add(record.pnumber)
        res.records[record.pnumber] = record
        i += 3

    return res



def averageTime(data, common):
    res = 0.0
    counter = 0
    for pnumber in common:
        res += data.records[pnumber].time
        counter += 1

    return res/ counter



def solvedDiffernece(result1, result2):
    data1 = parseData(result1)
    data2 = parseData(result2)
    has1_not2 = data1.solved - data2.solved
    has2_not1 = data2.solved - data1.solved
    print(has1_not2, " is only solved by method1")
    print(has2_not1, " is only solved by method2")
    common = data1.solved & data2.solved
    print("In the common set    ", common, ":")
    print("method 1 used ", averageTime(data1, common), " seconds for each file\n"
          "method 2 used ", averageTime(data2, common), " seconds for each file.")


def counterPassAnalyse(pather):
    data1 = parseData(pather)
    #pnumbers_notSolved = []
    counter_notpassingcounter = 0
    counter_all = 0
    for rec in data1.records:
        if rec not in data1.solved:
            #pnumbers_notSolved.append(rec)
            if data1.records[rec].z3_quries / float(data1.records[rec].candidates) < 0.2: # we consider it not passing
                counter_notpassingcounter += 1
            counter_all += 1

    print("In summary:  ", "total:  ", counter_all, "; not passing: ", counter_notpassingcounter)




if __name__ == "__main__":
    path1 = r"G:\reproduce\code2inv-templete-unsatcore-master\code2inv\prog_generator\testOn+Param+NoCounter+ordered+reward-5.2+after+totalRandom.txt"
    path2 = r"G:\reproduce\code2inv-templete-unsatcore-master\code2inv\prog_generator\testOn+Param+NoCounter+ordered+reward-5.2+after.txt"
    solvedDiffernece(path1, path2)


# if __name__ == "__main__":
#     pather = r"G:\reproduce\code2inv-templete-unsatcore-master\code2inv\prog_generator\testOn+Param+templete_const_smt_solving+ordered+reward-5.2+VarElimation+cold_start.txt"
#     counterPassAnalyse(pather)