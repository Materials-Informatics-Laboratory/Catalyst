import math
def get_distribution(params):
    if params['dist_type'] == 'exp':
        data = []
        for i in range(params['size']):
            val = params['vars'][0]*math.exp(-params['vars'][1]*i)
            if val < params['floor']:
                val = params['floor']
            data.append(val)
        return data











