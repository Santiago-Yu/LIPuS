


if __name__ == "__main__":
    result_file = open(r"G:\workspace\PrunningRL\Result\RL_Prunning_Final5_600s_1e7_random.txt", "r")
    lines = result_file.readlines()
    res = []
    for l in lines:
        if "None" in l:
            res.append("None")
        else:
            try:
                a1,a2 = l.split('	')
                time = float(a1)
                res.append(time)
            except:
                print(l)
                continue
    res = [str(x) for x in res]
    print("**************************************************")
    print('\n'.join(res))
