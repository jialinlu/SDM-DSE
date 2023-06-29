# Some useful functions

import os
import copy
import random
import string
import numpy as np
import time
import random
import igraph
import pygraphviz as pgv
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from simulink_model_generate import simulink_model

# GLOBAL VARIABLES
ORDER_NUM = 4
ABC_UB    = 2
ABC_LB    = 0.1
G_UB      = 1
G_LB      = 0.001
GBW_UB    = 10
GBW_LB    = 1

def gen_random_id():
    random_id = ''.join(random.sample(string.ascii_letters + string.digits, 6))
    return random_id

def save_res(head):
    os.chdir('./results')
    design_id = gen_random_id()
    file_name = head+'_'+design_id
    os.system('mkdir {}'.format(file_name))
    os.chdir(file_name)
    os.system('cp ../../adc_* ./')
    os.system('cp ../../sim_res* ./')
    os.system('cp ../../*png ./')
    os.system('cp ../../*.pkl ./')
    os.system('cp ../../*.py ./')
    os.chdir('../..')
    return design_id

class my_db():
    def __init__(self, x_columns, y_columns):
        super(my_db, self).__init__()

        if not os.path.exists('./database/'):
            os.mkdir('./database/')

        self.x_columns   = x_columns
        self.y_columns   = y_columns
        self.res_columns = x_columns + y_columns
        self.res_db      = pd.DataFrame(columns=self.res_columns)
        time_array       = time.localtime()
        self.t_str       = time.strftime("%Y_%m_%d_%H_%M",time_array)

    def wrapper(self, topo_params:pd.DataFrame, factor_params:pd.DataFrame):
        topo_params   = topo_params.reset_index(drop=True)
        factor_params = factor_params.reset_index(drop=True)
        params = pd.concat([topo_params,factor_params],axis=1)
        return params

    def update(self, x_df:pd.DataFrame, y:np.array):
        y_df   = pd.DataFrame(y, index=[0], columns=self.y_columns)
        res_df = self.wrapper(x_df, y_df)
        self.res_db = self.res_db.append(res_df, ignore_index=True)

    def save(self,type):
        if type == 'csv':
            save_name  = './database/res_'+self.t_str+'.csv'
            self.res_db.to_csv(save_name)
        elif type == 'html':
            save_name  = './database/res_'+self.t_str+'.html'
            self.res_db.to_html(save_name)
        elif type == 'excel':
            save_name  = './database/res_'+self.t_str+'.xls'
            self.res_db.to_excel(save_name)
        else:
            print('unsupport data type!!!')
            save_name = 'error'
        return save_name
    
    def check(self,new_topo,topo_ns,y_ns):
        check_list = []
        for i in range(self.res_db.shape[0]):
            fuck = 1
            q = self.res_db.iloc[i]
            for n in topo_ns:
                if new_topo.iloc[0][n] != q[n]:
                    fuck = 0
                    break
            if fuck == 1:
                check_list.append(i)
                #check_y = self.res_db.iloc[i,-1*len(y_ns):].to_frame().T
                #break
        if (check_list != []):
            best_fom = 0
            best_f   = check_list[0]
            for f in check_list:
                if (self.res_db['ic1'].iloc[f]<0) and (self.res_db['ic2'].iloc[f]<0) and (self.res_db['ic3'].iloc[f]<0) and (self.res_db['ic4'].iloc[f]<0) and (self.res_db['sndr'].iloc[f]<0):
                    tmp_fom = self.res_db['FOMs'].iloc[f]*(-1)
                    if tmp_fom > best_fom:
                        best_fom = tmp_fom
                        best_f   = f
            check_y = self.res_db.iloc[best_f,-1*len(y_ns):].to_frame().T
            return True, np.array(check_y,dtype=float)
        else:
            return False, 0

    def plot(self,save_pdf=False):
        data = self.res_db
        s = []
        sndr = 0
        f = []
        foms = 0
        for i in range(len(data)):
            d = data.iloc[[i]]
            if (d['ic1'].iloc[0]<0) and (d['ic2'].iloc[0]<0) and (d['ic3'].iloc[0]<0) and (d['ic4'].iloc[0]<0) and (d['sndr'].iloc[0]<0):
                tmp_f = d['FOMs'].iloc[0]*(-1)
                if tmp_f>foms:
                    foms = tmp_f
                tmp_s = d['sndr'].iloc[0]*(-1)
                if tmp_s>sndr:
                    sndr = tmp_s
            s.append(sndr)
            f.append(foms)

        fig = plt.figure()
        num_points = len(data)
        plt.plot(range(1, num_points+1), f, label='FOMs')
        plt.plot(range(1, num_points+1), s, label='SNDR')
        plt.xlabel('Simulations Times')
        plt.ylabel('Results')
        plt.legend()
        plt.savefig('./database/res_{}.png'.format(self.t_str))
        if save_pdf:
            plt.savefig('./database/res_{}.pdf'.format(self.t_str))

def sample_graph(points=ORDER_NUM+3,d=True,symmetry=True,return_params=False):
    '''
    the space of the graph is actually defined here
    order = 4
    generate a graph randomlly with some fix vertices/edges, and mutative edges
    the symmetry options of 'a' and 'b'
    '''
    order_num     = points-3
    topo_params   = []
    factor_params = []
    sweep_names   = []

    # init a DAG
    g = igraph.Graph(directed=d)

    ## add vertices
    g.add_vertices(points)
    g.vs[0]['type']  = 1 # vin
    g.vs[0]['name']  = 'vin'
    g.vs[points-1]['type']  = 2 # dac
    g.vs[points-1]['name']  = "dac"
    g.vs[points-2]['type']  = 3 # quantizer
    g.vs[points-2]['name']  = "quantizer"
    for i in range(1,order_num+1):
        g.vs[i]['type']  = 4 # integrator
        g.vs[i]['name']  = "integrator"
        #g.vs[i+1]['alfa']  = (1e4-1)/1e4         # init the alfa of integrator
        g.vs[i]['gbw']   = random.randint(GBW_LB, GBW_UB) # init the gbw of integrator, n([1,10]) * Fs
        factor_params.append({'name':'gbw'+str(i), 'type':'num', 'lb':GBW_LB, 'ub':GBW_UB})
        sweep_names.append('gbw'+str(i))

    '''add egdes randomly'''
    '''1. fixed egdes'''
    # between integrators and quantizer, type as 'c'
    edge_pairs = [[1,2],[2,3],[3,4],[4,5],[5,6]]
    for i,e in enumerate(edge_pairs):
        if i == 5:
            g.add_edge(e[0],e[1],weight=1,type=1)
        elif i== 4:
            factor_params.append({'name':'c_w_'+str(i), 'type':'num', 'lb':G_LB, 'ub':G_UB})
            sweep_names.append('c_w_'+str(i))
            g.add_edge(e[0],e[1],weight=random.uniform(G_LB, G_UB),type=1) # init the weights of Gain block
        else:
            factor_params.append({'name':'c_w_'+str(i), 'type':'num', 'lb':ABC_LB, 'ub':ABC_UB})
            sweep_names.append('c_w_'+str(i))
            g.add_edge(e[0],e[1],weight=random.uniform(ABC_LB, ABC_UB),type=1) # init the weights of Gain block
    '''2. edges to be tuned'''
    '''
    here I use a symmetric variable to select the generation of a and b
    '''
    if not symmetry:
        # add 'b' and 'a' freely
        # between vin and other vertices, type as 'b'
        for iv in range(2,points-1):
            topo_params.append({'name':'b_t_'+str(iv), 'type':'bool'})
            factor_params.append({'name':'b_w_'+str(iv), 'type':'num', 'lb':ABC_LB, 'ub':ABC_UB})
            sweep_names.append('b_w_'+str(iv))
            tmp = random.random()
            if tmp>0.5:
                g.add_edge(0,iv,weight=round(random.uniform(ABC_LB, ABC_UB),3),type=1)
            else:
                g.add_edge(0,iv,weight=0,type=0)
        # connect vin to integrator_1
        factor_params.append({'name':'b_w_'+str(1), 'type':'num', 'lb':ABC_LB, 'ub':ABC_UB})
        sweep_names.append('b_w_'+str(1))
        g.add_edge(0,1,weight=round(random.uniform(ABC_LB, ABC_UB),3),type=1)
        # forbid vin to dac
        g.add_edge(0,points-1,weight=0,type=0)
        
        # between dac and other veryices, type as 'a'
        for dv in range(2,order_num+1):
            topo_params.append({'name':'a_t_'+str(dv), 'type':'bool'})
            factor_params.append({'name':'a_w_'+str(dv), 'type':'num', 'lb':ABC_LB, 'ub':ABC_UB})
            sweep_names.append('a_w_'+str(dv))
            tmp = random.random()
            if tmp>0.5:
                g.add_edge(points-1,dv,weight=round(random.uniform(ABC_LB, ABC_UB),3),type=1)
            else:
                g.add_edge(points-1,dv,weight=0,type=0)
        # connect dac to integrator_1
        factor_params.append({'name':'a_w_'+str(1), 'type':'num', 'lb':ABC_LB, 'ub':ABC_UB})
        sweep_names.append('a_w_'+str(1))
        g.add_edge(points-1,1,weight=round(random.uniform(ABC_LB, ABC_UB),3),type=1)
    else:
        # 'a' and 'b' are paired
        for iv in range(2,points-2):
            topo_params.append({'name':'ab_t_'+str(iv), 'type':'bool'})
            factor_params.append({'name':'ab_w_'+str(iv), 'type':'num', 'lb':ABC_LB, 'ub':ABC_UB})
            sweep_names.append('ab_w_'+str(iv))
            tmp = random.random()
            if tmp>0.5:
                g.add_edge(0,iv,weight=round(random.uniform(ABC_LB, ABC_UB),3),type=1)
                g.add_edge(points-1,iv,weight=round(random.uniform(ABC_LB, ABC_UB),3),type=1)
            else:
                g.add_edge(0,iv,weight=0,type=0)
                g.add_edge(points-1,iv,weight=0,type=0)
        # connect vin to integrator_1 and dac to integrator_1
        factor_params.append({'name':'ab_w_'+str(1), 'type':'num', 'lb':ABC_LB, 'ub':ABC_UB})
        sweep_names.append('ab_w_'+str(1))
        random_w = round(random.uniform(ABC_LB, ABC_UB),3)
        g.add_edge(0,1,weight=random_w,type=1)
        g.add_edge(points-1,1,weight=random_w,type=1)
        # connect vin and quantizer
        topo_params.append({'name':'b_t_0_'+str(points-2), 'type':'bool'})
        factor_params.append({'name':'b_w_0_'+str(points-2), 'type':'num', 'lb':ABC_LB, 'ub':ABC_UB})
        sweep_names.append('b_w_0_'+str(points-2))
        tmp = random.random()
        if tmp>0.5:
            g.add_edge(0,points-2,weight=round(random.uniform(ABC_LB, ABC_UB),3),type=1)
        else:                                                                           
            g.add_edge(0,points-2,weight=0,type=0)
        # forbid vin to dac
        g.add_edge(0,points-1,weight=0,type=0)

    # between the intermediate node, if forward type as 'gb', if backward type as 'ga'
    for i in range(2,order_num+1):
        for mv in range(1,order_num):
            if ((mv > i+1) or (mv < i-1)) and (mv not in g.neighbors(i)):
                # gain_type = 'gb' or 'ga'
                if mv>(i+1):
                    topo_params.append({'name':'gb_t_'+str(i)+'_'+str(mv), 'type':'bool'})
                    factor_params.append({'name':'gb_w_'+str(i)+'_'+str(mv), 'type':'num', 'lb':G_LB, 'ub':G_UB})
                    sweep_names.append('gb_w_'+str(i)+'_'+str(mv))
                else:
                    topo_params.append({'name':'ga_t_'+str(i)+'_'+str(mv), 'type':'bool'})
                    factor_params.append({'name':'ga_w_'+str(i)+'_'+str(mv), 'type':'num', 'lb':G_LB, 'ub':G_UB})
                    sweep_names.append('ga_w_'+str(i)+'_'+str(mv))
                tmp = random.random()
                if tmp>0.5:
                    g.add_edge(i,mv,weight=round(random.uniform(G_LB, G_UB),3),type=1)
                else:
                    g.add_edge(i,mv,weight=0,type=0)
            else:
                pass
    # between integrators and quantizer
    for i in range(1,order_num):
        # this gain type is 'gb'
        topo_params.append({'name':'gb_t_'+str(i)+'_5', 'type':'bool'})
        factor_params.append({'name':'gb_w_'+str(i)+'_5', 'type':'num', 'lb':G_LB, 'ub':G_UB})
        sweep_names.append('gb_w_'+str(i)+'_5')
        if tmp>0.5:
            g.add_edge(i,mv,weight=round(random.uniform(G_LB, G_UB),3),type=1)
        else:
            g.add_edge(i,mv,weight=0,type=0)

    if return_params:
        return topo_params, factor_params, sweep_names
    else:
        return g

def parser_param(param,points=ORDER_NUM+3,d=True,symmetry=True):
    '''parser the param vector given by our bo engion and generate the graph'''
    order_num = points-3
    # init a DAG
    g = igraph.Graph(directed=d)
    ## add vertices
    g.add_vertices(points)
    g.vs[0]['type']  = 1 # vin
    g.vs[0]['name']  = 'vin'
    g.vs[points-1]['type']  = 2 # dac
    g.vs[points-1]['name']  = "dac"
    g.vs[points-2]['type']  = 3 # quantizer
    g.vs[points-2]['name']  = "quantizer"
    for i in range(1,order_num+1):
        g.vs[i]['type']  = 4 # integrator
        g.vs[i]['name']  = "integrator"
        #g.vs[i+1]['alfa']  = (1e4-1)/1e4
        g.vs[i]['gbw']   = param['gbw'+str(i)].iloc[0]

    '''add egdes randomly'''
    '''1. fixed egdes'''
    # between integrators and quantizer, type as 'c'
    edge_pairs = [[1,2],[2,3],[3,4],[4,5],[5,6]]
    for i,e in enumerate(edge_pairs):
        if i == 5:
            g.add_edge(e[0],e[1],weight=1,type=1)
        else:
            g.add_edge(e[0],e[1],weight=param['c_w_'+str(i)].iloc[0],type=1)
    '''2. edges to be tuned'''
    '''
    here I use a symmetric variable to select the generation of a and b
    '''
    if not symmetry:
        # add 'b' and 'a' freely
        # between vin and other vertices, type as 'b'
        for iv in range(2,points-1):
            g.add_edge(0,iv,weight=param['b_w_'+str(iv)].iloc[0],type=int(param['b_t_'+str(iv)].iloc[0]))
        g.add_edge(0,1,weight=param['b_w_'+str(1)].iloc[0],type=1) # connect vin to integrator_1
        g.add_edge(0,points-1,weight=0,type=0) # forbid vin to dac
        # between dac and other veryices, type as 'a'
        for dv in range(2,order_num+1):
            g.add_edge(points-1,dv,weight=param['a_w_'+str(dv)].iloc[0],type=int(param['a_t_'+str(dv)].iloc[0]))
        g.add_edge(points-1,1,weight=param['a_w_'+str(1)].iloc[0],type=1) # connect dac to integrator_1
    else:
        # 'a' and 'b' are paired
        for iv in range(2,points-2):
            g.add_edge(0,iv,weight=param['ab_w_'+str(iv)].iloc[0],type=int(param['ab_t_'+str(iv)].iloc[0]))
            g.add_edge(points-1,iv,weight=param['ab_w_'+str(iv)].iloc[0],type=int(param['ab_t_'+str(iv)].iloc[0]))
        # connect vin to integrator_1 and dac to integrator_1
        g.add_edge(0,1,weight=param['ab_w_'+str(1)].iloc[0],type=1)
        g.add_edge(points-1,1,weight=param['ab_w_'+str(1)].iloc[0],type=1)
        # connect vin and quantizer
        g.add_edge(0,points-2,weight=param['b_w_0_'+str(points-2)].iloc[0],type=int(param['b_t_0_'+str(points-2)].iloc[0]))
        # forbid vin to dac
        g.add_edge(0,points-1,weight=0,type=0)

    # between the intermediate node, if forward type as 'gb', if backward type as 'ga'
    for i in range(2,order_num+1):
        for mv in range(1,order_num):
            if ((mv > i+1) or (mv < i-1)) and (mv not in g.neighbors(i)):
                # gain_type = 'gb' or 'ga'
                if mv>(i+1):
                    g.add_edge(i,mv,weight=param['gb_w_'+str(i)+'_'+str(mv)].iloc[0],type=int(param['gb_t_'+str(i)+'_'+str(mv)].iloc[0]))
                else:
                    g.add_edge(i,mv,weight=param['ga_w_'+str(i)+'_'+str(mv)].iloc[0],type=int(param['ga_t_'+str(i)+'_'+str(mv)].iloc[0]))
            else:
                pass
    # between integrators and quantizer
    for i in range(1,order_num):
        # this gain type is 'gb'
        g.add_edge(i,5,weight=param['gb_w_'+str(i)+'_5'].iloc[0],type=int(param['gb_t_'+str(i)+'_5'].iloc[0]))
    return g

def add_node(graph, g, node_id, label, node_num, shape='box'):
    if node_id == 0:
        label = 'Vin'
    elif node_id == node_num:
        label = 'yout'
    elif node_id == node_num-1:
        label = 'dac'
    elif node_id == node_num-2:
        label = 'sum & quantizer'
    else:
        label = 'sum & integrator '+str(node_id)+': '+ str(g.vs[node_id]['gbw']) 
    label = f"{label}"
    graph.add_node(node_id, label=label,shape=shape,fontsize=24)

def draw_network(g, path, backbone=False):
    graph = pgv.AGraph(directed=True, strict=True, fontname='Helvetica', arrowtype='open')
    # add vertex
    for idx in range(g.vcount()):
        add_node(graph, g, idx, g.vs[idx]['name'], g.vcount())
    add_node(graph, g, g.vcount(), 'yout', g.vcount()) # add yout node
    # add edge
    for idx in range(g.vcount()):
        for node in g.successors(idx):
            eid = g.get_eid(idx,node)
            if g.es[eid]['type'] != 0:
                edge_label = str(round(g.es['weight'][eid],3))+','+str(g.es['type'][eid])
                graph.add_edge(idx, node, label=edge_label)
    graph.add_edge(g.vcount()-1, g.vcount()) # connect yout

    graph.layout(prog='dot')
    graph.draw(path)

def plot_DAG(g, res_dir, name, backbone=False, pdf=False):
    # backbone: puts all nodes in a straight line
    file_name = os.path.join(res_dir, name+'.png')
    if pdf:
        file_name_pdf = os.path.join(res_dir, name+'.pdf')
        draw_network(g, file_name_pdf, backbone)
    draw_network(g, file_name, backbone)
    return file_name

def get_adc_param(param):
    # get osr, alfa, ncomparatori from X
    if ('osr' in param.columns):
        osr = param['osr'].iloc[0]
    else:
        osr = 256*4

    if ('alfa' in param.columns):
        alfa = param['alfa'].iloc[0]
    else:
        alfa = (1e4-1)/1e4

    return osr, alfa

def run_simulation(sweep_names, param, dag, matlab_eng, symmetry, test):
    #os.chdir('./tmp_circuits')
    os.system('source ./clear.sh')

    goal = []
    cons = []

    osr, alfa = get_adc_param(param)
    ncomparatori = 15

    model = simulink_model(dag, osr, alfa, ncomparatori)
    sndr, enob, power, fom, ic1, ic2, ic3, ic4 = model.run(matlab_eng)

    g = -1*sndr - fom # consern both sndr and power
    goal.append(g)

    cons.append(-1*sndr)  # sndr>std_sndr
    cons.append(-1*enob)  # enob>std_enob
    cons.append(-1*power) # power constr, to be determinted
    cons.append(-1*fom)   # fom constr, >185 will be excellent
    cons.append(ic1-0.8)  # integrators output should smaller then 0.8
    cons.append(ic2-0.8)
    cons.append(ic3-0.8)
    cons.append(ic4-1.1)

    #os.chdir('..')

    return goal, cons

def sweep_factor(design_id,og,param,sweep_names,sndr,matlab_eng,symmetry,plot,save_pdf):
    factor_robust = 0
    sweep_sndrs = []

    # sweep gbw and abcg, +- 20%
    for s in range(-20,20):
        print('sweep factor {}-th'.format(s))
        os.system('source ./clear.sh')

        param_sm = copy.copy(param)
        for gbw in range(og.vcount()-3):
            param_sm[sweep_names[gbw]] = param[sweep_names[gbw]]*(1+s/100)
        for abcg in range(og.vcount()-3,len(sweep_names)):
            param_sm[sweep_names[abcg]] = param[sweep_names[abcg]]*(1+s/100)
        
        dag_sm = parser_param(param_sm)
        osr, alfa = get_adc_param(param_sm)
        ncomparatori = 15
        model_sm = simulink_model(dag_sm, osr, alfa, ncomparatori)
        tmp_sndr, _, _, _, _, _, _, _ = model_sm.run(matlab_eng)
        sweep_sndrs.append(tmp_sndr)
        if abs(tmp_sndr-sndr) <= 5:
            factor_robust += 1
    
    # plot the sweep results
    if plot:
        fig = plt.figure()
        num_points = len(sweep_sndrs)
        plt.plot(range(-20, 20), sweep_sndrs)
        plt.axhline(y=sndr+5, ls="-", c="red")
        plt.axhline(y=sndr-5, ls="-", c="red")
        plt.xlabel('sweep factor percent')
        plt.ylabel('SNDR')
        plt.legend()
        plt.savefig('./results_select/4th_order_{}/sweep_factor_{}.png'.format(design_id,design_id))
        if save_pdf:
            plt.savefig('./results_select/4th_order_{}/sweep_factor_{}.pdf'.format(design_id,design_id))

    return factor_robust

def sweep_ncomparatori(design_id,param,sndr,matlab_eng,symmetry,plot,save_pdf):
    ncomparatori_robust = 0
    sweep_sndrs = []

    # sweep gbw and abcg, +- 20%
    for ncomparatori in range(15,0,-1):
        print('sweep ncomparatori {}-th'.format(ncomparatori))
        os.system('source ./clear.sh')
        dag_sm = parser_param(param)
        osr, alfa = get_adc_param(param)
        sm = simulink_model(dag_sm, osr, alfa, ncomparatori)
        tmp_sndr, _, _, _, _, _, _, _ = sm.run(matlab_eng)
        sweep_sndrs.append(tmp_sndr)
        if abs(tmp_sndr-sndr) <= 5:
            ncomparatori_robust += 1
    
    # plot the sweep results
    if plot:
        fig = plt.figure()
        num_points = len(sweep_sndrs)
        plt.plot(range(15, 0, -1), sweep_sndrs)
        plt.axhline(y=sndr+5, ls="-", c="red")
        plt.axhline(y=sndr-5, ls="-", c="red")
        plt.xlabel('sweep ncomparatori')
        plt.ylabel('SNDR')
        plt.legend()
        plt.savefig('./results_select/4th_order_{}/sweep_ncomparatori_{}.png'.format(design_id,design_id))
        if save_pdf:
            plt.savefig('./results_select/4th_order_{}/sweep_ncomparatori_{}.pdf'.format(design_id,design_id))

    return ncomparatori_robust

def robust_analysis(design_id,og,param,sweep_names,sndr,matlab_eng,symmetry,plot=False,save_pdf=False):
    factor_robust = sweep_factor(design_id,og,param,sweep_names,sndr,matlab_eng,symmetry,plot,save_pdf)
    ncomparatori_robust = sweep_ncomparatori(design_id,param,sndr,matlab_eng,symmetry,plot,save_pdf)
    return factor_robust + ncomparatori_robust
