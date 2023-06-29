# sigma-delta adc loop topology optimization with HEBO: https://github.com/huawei-noah/HEBO
# optimize topology and device parameters asynchronously
# Date: 2022/03/04
# Version: 0.0.1

import matlab.engine
import os
import copy 
import pickle
import time
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

from hebo.design_space.design_space import DesignSpace
from hebo.optimizers.general import GeneralBO
from hebo.optimizers.hebo import HEBO

from utils import *

import warnings
warnings.filterwarnings("ignore")

#####################################################################################
############################## OPTIMIZATION SETUP ###################################
#####################################################################################
# problem formulation
obj_nums    = 1
constr_nums = 8
SYMMETRY    = True
std_sndr  = 122.06 # sndr should >= nbit*6.02+1.76 + 30dB, here is 122.06
std_enob  = (std_sndr-1.76)/6.02

# optimization parameters
bo_algorithm       = 'general' # 'general' or 'mace'
topo_sample_nums   = 5
factor_sample_nums = 10
topo_iter_nums     = 100
factor_iter_nums   = 50
factor_break_nums  = 30
para_nums = 1

# get the params from defined graph
topo_params, factor_params,sweep_names = sample_graph(symmetry=SYMMETRY, return_params=True)
topo_space    = DesignSpace().parse(topo_params)
factor_space  = DesignSpace().parse(factor_params)
topo_names    = topo_space.numeric_names
factor_names  = factor_space.numeric_names

# define extra adc params
osr_param  = {'name':'osr', 'type':'num', 'lb':16, 'ub':128}
factor_params.append(osr_param)
alfa_param = {'name':'alfa', 'type':'num', 'lb':((100-1)/100), 'ub':((10000-1)/10000)}
factor_params.append(alfa_param)
# ncomparatori_param = {'name':'ncomparatori', 'type':'num', 'lb':1, 'ub':15}
# all_params.append(ncomparatori_param)

print(topo_names)
print(factor_names)
print('{} topology and {} size parameters have been parsered !'.format(len(topo_names),len(factor_names)))

# init the database
x_col  = topo_names+factor_names
y_col  = ['fom','sndr','enob', 'power','FOMs','ic1','ic2','ic3','ic4']
res_db = my_db(x_col, y_col)
print('all simulation data will be saved to {}'.format(res_db.t_str))

# start the matlab engine
val=os.system("source ./start_matlab.sh")
eng = matlab.engine.connect_matlab()
print('MATLAB engine started !')

# wrapper the objective function
def wrapper_topo_factor(topo_params:pd.DataFrame, factor_params:pd.DataFrame):
    topo_params   = topo_params.reset_index(drop=True)
    factor_params = factor_params.reset_index(drop=True)
    params = pd.concat([topo_params,factor_params],axis=1)
    return params

def ADC(params : pd.DataFrame) -> np.ndarray:
    batch_size = params.iloc[:,0].size
    Os  = []
    Cs  = []
    for i in range(batch_size):
        param = params[i:i+1]
        dag = parser_param(param, symmetry=SYMMETRY)
        o, c = run_simulation(sweep_names, param, dag, eng, symmetry=SYMMETRY, test=False)
        Os.append(o)
        Cs.append(c)
    Os = np.array(Os).reshape(batch_size, obj_nums)
    Cs = np.array(Cs).reshape(batch_size, constr_nums)
    return np.hstack([Os,Cs])

#####################################################################################
########################### OPTIMIZATION BEGIN HERE #################################
#####################################################################################
'''use two-level bo to optimize topo and factor separately'''
# init the upper level BO model
if bo_algorithm == 'general':
    print('using general BO algorithm for the upper level optimization')
    topo_opt = GeneralBO(space=topo_space, num_obj=obj_nums, num_constr=constr_nums, rand_sample=topo_sample_nums)
elif bo_algorithm == 'mace':
    print('using HEBO algorithm for the upper level optimization')
    topo_opt = HEBO(topo_space, model_name = 'gpy', rand_sample = topo_sample_nums)
else:
    print('unsupport BO algorithm ...')

cnvg_times = 0
best_fom   = 0
best_sndr  = 0
best_enob  = 0
best_power = 0
best_F     = 0
best_ic1   = 0
best_ic2   = 0
best_ic3   = 0
best_ic4   = 0
best_design_id = 'null'
sul_nums = 0
best_y   = 0
best_x   = 0

flow_begin = time.time()
for u in range(1,topo_iter_nums+1):

    topo_suggest = topo_opt.suggest(n_suggestions=para_nums)
    check_res, check_y = res_db.check(topo_suggest, topo_names, y_col)

    if check_res:
        print('an old one~~~')
        y_suggest = check_y
    else:
        # init the lower level BO model
        factor_opt = GeneralBO(space=factor_space, num_obj=obj_nums, num_constr=constr_nums, rand_sample=factor_sample_nums)
        factor_sul_nums = 0
        tmp_best_y      = np.random.rand(obj_nums+constr_nums).reshape(1,obj_nums+constr_nums)
        tmp_best_x      = 0
        tmp_best_fom    = 0
        bak_fom  = 0
        bak_y    = np.random.rand(obj_nums+constr_nums).reshape(1,obj_nums+constr_nums)

        for l in range(1,factor_iter_nums+1):

            factor_suggest = factor_opt.suggest(n_suggestions=para_nums)
            params = wrapper_topo_factor(topo_suggest,factor_suggest)
            tmp_y_suggest = ADC(params)
            factor_opt.observe(factor_suggest,tmp_y_suggest)
            tmp_fom = tmp_y_suggest[0][0]

            # update results database
            res_db.update(params,tmp_y_suggest)
            save_db = res_db.save(type='html')

            # track the fom value during the lower level 
            if (tmp_fom < tmp_best_fom) and (tmp_y_suggest[0][1]<0) and (tmp_y_suggest[0][2]<0) and (tmp_y_suggest[0][5]<0) and (tmp_y_suggest[0][6]<0) and (tmp_y_suggest[0][7]<0) and (tmp_y_suggest[0][8]<0) :
                factor_sul_nums += 1
                sul_nums += 1

                tmp_best_fom    = tmp_fom
                tmp_best_factor = factor_suggest
                tmp_best_y = tmp_y_suggest
                tmp_best_x = params

                # save results
                with open('./x_opt.pkl','wb') as xp:
                    pickle.dump(tmp_best_x,xp)
                p = plot_DAG(parser_param(tmp_best_x), './', 'g_opt', backbone=False, pdf=False)
                design_id = save_res('4th_order')

            # for bo observe while constrs not satisfied
            if tmp_fom < bak_fom:
                bak_fom = tmp_fom
                bak_y   = tmp_y_suggest

            # track the fom value during the upper level 
            if (tmp_y_suggest[0][0] < best_fom) and (tmp_y_suggest[0][1]<0) and (tmp_y_suggest[0][2]<0) and (tmp_y_suggest[0][5]<0) and (tmp_y_suggest[0][6]<0) and (tmp_y_suggest[0][7]<0) and (tmp_y_suggest[0][8]<0) :
                best_y = tmp_y_suggest
                best_x = params
                best_fom   = tmp_y_suggest[0][0]
                best_sndr  = -1*tmp_y_suggest[0][1]
                best_enob  = -1*tmp_y_suggest[0][2]
                best_power = -1*tmp_y_suggest[0][3]
                best_F     = -1*tmp_y_suggest[0][4]
                best_ic1   = tmp_y_suggest[0][5] + 0.8
                best_ic2   = tmp_y_suggest[0][6] + 0.8
                best_ic3   = tmp_y_suggest[0][7] + 0.8
                best_ic4   = tmp_y_suggest[0][8] + 1.1
                
                best_design_id = design_id
                cnvg_times = len(res_db.res_db)+1

            # while topo is bad, break
            if (l>factor_break_nums) and (factor_sul_nums==0):
                break

        if factor_sul_nums>0:
            print('find {} solutions in {}-th iter'.format(factor_sul_nums,u))
        else:
            tmp_best_y = bak_y
        y_suggest = tmp_best_y

    print('iter {}: fom={}, sndr={} dB, enob={} bit, power={} pW, FOMs={}'.format(u,-1*y_suggest[0][0],-1*y_suggest[0][1],-1*y_suggest[0][2],-1*y_suggest[0][3],-1*y_suggest[0][4]))
    print('intergrator output:{} {} {} {}'.format(y_suggest[0][5]+0.8,y_suggest[0][6]+0.8,y_suggest[0][7]+0.8,y_suggest[0][8]+1.1))
    print('   ')

    if bo_algorithm == 'general':
        topo_opt.observe(topo_suggest,y_suggest)
    elif bo_algorithm == 'mace':
        topo_opt.observe(topo_suggest,y_suggest[:,0].reshape(len(y_suggest),1))
    else:
        print('unsupport BO algorithm ...')
    
    if u % 10 == 0:
        print('Total iter %d: best_fom = %.2f, best_sndr = %.2f, best_enob = %.2f, best_FOMs = %.2f' % (u, best_fom, best_sndr, best_enob, best_F))
        print('the best integrator outputs are: %.2f, %.2f, %.2f, %.2f' % (best_ic1,best_ic2,best_ic3,best_ic4))
        print('current best adc id is {} which was found in {}-th simulation'.format(best_design_id, cnvg_times))
        print('*'*100)
flow_end = time.time()

print('full optimization finished, total {} simulations which cost {} hours'.format(len(res_db.res_db), round((flow_end-flow_begin)/3600),2))
print('find {} solutions at {}-th tries (convergence times)'.format(sul_nums,cnvg_times))
print('the best adc loop topology result is:')
print(best_y)
print(best_x)

#####################################################################################
#################################### CLEAN UP #######################################
#####################################################################################
if sul_nums > 0:
    os.system('cp -r ./results/4th_order_{} ./results_select/'.format(best_design_id))
    print('best adc design {} have been saved to results_select folder'.format(best_design_id))
    # os.system('rm -rf ./results')
    # os.system('mkdir results')
val=os.system('source ./clear.sh')

'''end the matlab engine'''
eng.exit()
val=os.system('source ./kill_matlab.sh')
print('MATLAB engine killed !')

