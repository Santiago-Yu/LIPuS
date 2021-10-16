

def getParFromModule(moduler, prefix):
    res = {}
    id = 0
    keys = moduler.state_dict().keys()
    for x, keyname in zip(moduler.parameters(), keys):
        res[prefix+'_'+str(keyname)] = x
        id += 1
    return res


