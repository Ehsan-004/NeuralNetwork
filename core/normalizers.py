def normalize(d):
    dmin = min(d)
    drange = max(d) - dmin
    res = []
    for x in d:
        res.append((x-dmin) / drange)
    
    return res
    
    