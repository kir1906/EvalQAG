import json

def parse_json(temp):
    try:
        return json.loads(temp)
    except:
        return temp

def safe_int(s):
    try:
        return float(s.split(',')[0].split(':')[-1].strip())
    except:
        return -1
