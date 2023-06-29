# Generate behavioral models of sigma-delta ADC in matlab
# Implemented with SDToolbox2: https://ww2.mathworks.cn/matlabcentral/fileexchange/25811-sdtoolbox-2
# Date: 2021/11/04
# Version: 0.0.1

import os
import re
#import matlab.engine
import numpy as np
from utils import *

class simulink_model():
    def __init__(self, opt_dag, osr, alfa, ncomparatori):
        super(simulink_model, self).__init__()

        self.opt_dag = opt_dag
        self.opened_file = open('./adc_gen_sim.m','w')
        self.adder_inputs = {} # for routing with adders

        '''get the loop order'''
        self.order = self.opt_dag.vcount() - 3  # [3,7], the order number of the sigle loop

        #####################################################################################
        ############## SPECIFICATIONS (optimization goals and constrains) ################### 
        #####################################################################################
        self.sndr   = 0 # maximize 
        self.enob   = 0 # maximize
        self.power  = 0 # minimize
        self.robust = 0 # minimize
        
        self.std_sndr  = 137.06 # sndr should >= nbit*6.02+1.76 + 45dB, here is 137.06
        self.std_enob  = (self.std_sndr-1.76)/6.02
        self.std_fom   = 185    # fom > 185 will be excellent
        self.std_power = 1

        #####################################################################################
        ############ GLOBAL VARIABLES (simulation settings and fixed parameters) ############ 
        #####################################################################################
        '''simulation settings'''
        self.k    = 1.38e-23 # Boltzmann Constant
        self.Temp = 297      # Absolute Temperature in Kelvin
        self.bw   = 22.05e3  # Base-band
        self.osr  = osr      # oversampling rate, e.g. 256*4
        self.Fs   = self.osr*2*self.bw # oversampling frequency
        self.Ts   = 1/self.Fs
        self.N    = 4096*8   # samples number
        self.nper = 5
        self.Fin        = self.nper*self.Fs/self.N     # Input signal frequency (Fin = nper*Fs/N)
        self.finrad     = self.Fin*2*np.pi             # Input signal frequency in radians
        self.Ampl       = 0.9                          # Input signal amplitude [V]
        self.Ntransient = 50
        '''loop topology parameters'''
        self.nbit         = 4  # [1,6], the bit number of the quantizer
        self.ncomparatori = ncomparatori # 2^(nbit) -1, e.g. 15
        '''device parameters'''
        self.alfa  = alfa        # fix the Gian of integrators for now, e.g. (1e4-1)/1e4
        self.Cs    = 2.5e-12     # Integrating Capacitance of the first integrator
        self.Amax  = 1000        # Op-amp saturation value [V]
        self.sr    = 2000000e6   # Op-amp slew rate [V/s]
        self.delta = 0           # Random Sampling jitter (std. dev.) [s]
        self.match = 1e-13       # Realistic value, but not related to any technology (because of non disclosure agreement)

        #####################################################################################
        ############ OPTIMIZATION VARIABLES (circuit parameters to be tuned) ################
        #####################################################################################
        ''' 
        4th order as the example, constrct the single loop topology with the DAG
        classify all points to 3 types according their function: vin, adders and quantizer, ignore yout 
        '''
        self.num_point = self.order + 3 # with one vin, one quantizer, one dac
        self.num_opamp = self.order     # opamp(integrator) number is exact the order number

        '''device parameters number are controlled by topology parameters'''
        # abcg, the factors of "Gain" block

        '''to be supported in next version'''
        # alfa, Op-amp finite gain (alfa=(A-1)/A -> ideal op-amp alfa=1)

        # GBW, Op-amp GBW [Hz] = (1,10) * Fs

    def init_model(self):
        # init a script for simulink model and simulation
        self.opened_file.writelines('clear;\n')
        self.opened_file.writelines('warning(\'off\');\n')
        # new a system
        self.opened_file.writelines('new_system(\'adc_test\');\n')
        # insert input signal
        self.opened_file.writelines('add_block(\'template/Jittered SineWave\',\'adc_test/Jittered SineWave\');\n')
        self.opened_file.writelines('SineWave_port=get_param(\'adc_test/Jittered SineWave\',\'PortHandles\');\n')
        #self.opened_file.writelines('add_block(\'template/Switch Non-Linearity\',\'adc_test/Switch Non-Linearity\');\n')
        #self.opened_file.writelines('Switch_port=get_param(\'adc_test/Switch Non-Linearity\',\'PortHandles\');\n')
        # insert yout/psd/scope block
        self.opened_file.writelines('add_block(\'template/yout\',\'adc_test/yout\');\n')
        self.opened_file.writelines('yout_port=get_param(\'adc_test/yout\',\'PortHandles\');\n')
        self.opened_file.writelines('add_block(\'template/Power Spectral Density\',\'adc_test/Power Spectral Density\');\n')
        self.opened_file.writelines('psd_port=get_param(\'adc_test/Power Spectral Density\',\'PortHandles\');\n')
        self.opened_file.writelines('add_block(\'template/Scope1\',\'adc_test/Scope1\');\n')
        self.opened_file.writelines('Scope_port=get_param(\'adc_test/Scope1\',\'PortHandles\');\n')
        # insert ADC-DAC block
        self.opened_file.writelines('add_block(\'template/ ADC-DAC\',\'adc_test/ADC-DAC\');\n')
        self.opened_file.writelines('adda_port=get_param(\'adc_test/ADC-DAC\',\'PortHandles\');\n')
        # routing
        #self.opened_file.writelines('add_line(\'adc_test\',SineWave_port.Outport(1),Switch_port.Inport(1),\'autorouting\',\'on\');\n') # sinwave-->switch-->
        self.opened_file.writelines('add_line(\'adc_test\',adda_port.Outport(1),yout_port.Inport(1),\'autorouting\',\'on\');\n')       # -->ADC-DAC-->yout-->
        self.opened_file.writelines('add_line(\'adc_test\',adda_port.Outport(1),psd_port.Inport(1),\'autorouting\',\'on\');\n')        # -->ADC-DAC-->psd-->
        self.opened_file.writelines('add_line(\'adc_test\',adda_port.Outport(2),Scope_port.Inport(1),\'autorouting\',\'on\');\n')      # -->ADC-DAC-->Scope-->

    def insert_gainBlock(self,index,gain,type):
        if type == 'a':
            self.opened_file.writelines('add_block(\'template/c3\',\'adc_test/'+type+'_'+index+'\');\n')
            self.opened_file.writelines('set_param(\'adc_test/'+type+'_'+index+'\',\'Gain\',\''+str(gain)+'\');\n')
            self.opened_file.writelines(type+'_port_'+index+'=get_param(\'adc_test/'+type+'_'+index+'\',\'PortHandles\');\n')
        elif type == 'b':
            self.opened_file.writelines('add_block(\'template/c2\',\'adc_test/'+type+'_'+index+'\');\n')
            self.opened_file.writelines('set_param(\'adc_test/'+type+'_'+index+'\',\'Gain\',\''+str(gain)+'\');\n')
            self.opened_file.writelines(type+'_port_'+index+'=get_param(\'adc_test/'+type+'_'+index+'\',\'PortHandles\');\n')
        elif type == 'ga':
            self.opened_file.writelines('add_block(\'template/c4\',\'adc_test/'+type+'_'+index+'\');\n')
            self.opened_file.writelines('set_param(\'adc_test/'+type+'_'+index+'\',\'Gain\',\''+str(gain)+'\');\n')
            self.opened_file.writelines(type+'_port_'+index+'=get_param(\'adc_test/'+type+'_'+index+'\',\'PortHandles\');\n')
        else:
            self.opened_file.writelines('add_block(\'template/c1\',\'adc_test/'+type+'_'+index+'\');\n')
            self.opened_file.writelines('set_param(\'adc_test/'+type+'_'+index+'\',\'Gain\',\''+str(gain)+'\');\n')
            self.opened_file.writelines(type+'_port_'+index+'=get_param(\'adc_test/'+type+'_'+index+'\',\'PortHandles\');\n')

    def insert_realIntegrator(self,gbw,index):
        self.opened_file.writelines('add_block(\'template/REAL Integrator (with Delay)\',\'adc_test/REAL Integrator (with Delay)_'+str(index)+'\');\n')
        #self.opened_file.writelines('set_param(\'adc_test/REAL Integrator (with Delay)_'+str(index)+'\',\'FiniteGain\',\'(1e4-1)/1e4\');\n')
        self.opened_file.writelines('set_param(\'adc_test/REAL Integrator (with Delay)_'+str(index)+'\',\'GBW\',\''+str(gbw)+'\');\n')
        self.opened_file.writelines('interg_port_'+str(index)+'=get_param(\'adc_test/REAL Integrator (with Delay)_'+str(index)+'\',\'PortHandles\');\n')

    def insert_idealIntegrator(self,index):
        self.opened_file.writelines('add_block(\'template/IDEAL Integrator (with delay)\',\'adc_test/IDEAL Integrator (with delay)_'+str(index)+'\');\n')
        self.opened_file.writelines('interg_port_'+str(index)+'=get_param(\'adc_test/IDEAL Integrator (with delay)_'+str(index)+'\',\'PortHandles\');\n')

    def insert_adder(self,index,degree):
        self.opened_file.writelines('add_block(\'template/Sum of Elements\',\'adc_test/Sum of Elements_'+str(index)+'\');\n')
        signs = '+'*degree
        self.opened_file.writelines('set_param(\'adc_test/Sum of Elements_'+str(index)+'\',\'ListOfSigns\',\''+signs+'\');\n')
        self.opened_file.writelines('sum_port_'+str(index)+'=get_param(\'adc_test/Sum of Elements_'+str(index)+'\',\'PortHandles\');\n')
        return 'sum_port_'+str(index)
    
    def insert_simout(self, index):
        self.opened_file.writelines('add_block(\'template/yout\',\'adc_test/integrators_out_'+str(index)+'\');\n')
        self.opened_file.writelines('set_param(\'adc_test/integrators_out_'+str(index)+'\',\'VariableName\',\'integrators_out_'+str(index)+'\');\n')
        self.opened_file.writelines('integrators_out_'+str(index)+'_port=get_param(\'adc_test/integrators_out_'+str(index)+'\',\'PortHandles\');\n')
        return 'integrators_out_'+str(index)+'_port'

    def gen_mdoel(self):
        # following steps contain sequence relationship, please do not adjust
        self.opened_file.writelines('%%%%%%%%%%%%%%\n')
        self.opened_file.writelines('%%automatically generated by script\n')
        self.opened_file.writelines('\n')
        self.opened_file.writelines('%%%%%%%%%%%%%%\n')
        self.opened_file.writelines('%%build the loop topology based on optimized graph\n')
        # initialize the model, insert fixed blocks
        self.init_model()
        
        # insert all integrators and adders
        for i in range(1,self.order+2):
            if i<=self.order:
                '''real integrator'''
                #alfa_tmp = self.opt_dag.vs[i]['alfa']
                gbw_tmp  = self.opt_dag.vs[i]['gbw'] * self.Fs
                self.insert_realIntegrator(gbw_tmp,i)
                ''' ideal integrator '''
                #self.insert_idealIntegrator(i)
                deg = 0
                for node in self.opt_dag.predecessors(i):
                    eid = self.opt_dag.get_eid(node,i)
                    if self.opt_dag.es[eid]['type'] != 0:
                        deg += 1
                add_name = self.insert_adder(i,deg)
                self.adder_inputs.update({add_name:1})
                # routing, connect: -->adder-->integrator-->
                self.opened_file.writelines('add_line(\'adc_test\',sum_port_'+str(i)+'.Outport(1),interg_port_'+str(i)+'.Inport(1),\'autorouting\',\'on\');\n')
            else:
                deg = 0
                for node in self.opt_dag.predecessors(i):
                    eid = self.opt_dag.get_eid(node,i)
                    if self.opt_dag.es[eid]['type'] != 0:
                        deg += 1
                add_name = self.insert_adder(i,deg)
                self.adder_inputs.update({add_name:1})
                # routing, connect last adder and adda, -->adder-->ADC-DAC-->
                self.opened_file.writelines('add_line(\'adc_test\',sum_port_'+str(i)+'.Outport(1),adda_port.Inport(1),\'autorouting\',\'on\');\n')
        
        # insert all gain blocks, e.g. abcg
        # gain blocks between integrators and quantizer, type as 'c'
        for i in range(1,self.order+1):
            gain_type = 'c'
            eid  = self.opt_dag.get_eid(i,i+1)
            if self.opt_dag.es[eid]['type'] != 0:
                gain = self.opt_dag.es['weight'][eid]
                idx  = str(i)+'_'+str(i+1)
                self.insert_gainBlock(idx,gain,gain_type)
                # routing, connect: -->integrator-->gain-->adder-->integrator-->
                self.opened_file.writelines('add_line(\'adc_test\',interg_port_'+str(i)+'.Outport(1),'+gain_type+'_port_'+idx+'.Inport(1),\'autorouting\',\'on\');\n')
                self.opened_file.writelines('add_line(\'adc_test\','+gain_type+'_port_'+idx+'.Outport(1),sum_port_'+str(i+1)+'.Inport({}),\'autorouting\',\'on\');\n'.format(self.adder_inputs['sum_port_'+str(i+1)]))
                self.adder_inputs['sum_port_'+str(i+1)] = self.adder_inputs['sum_port_'+str(i+1)] + 1
                # routing, connect: -->integrator-->gain-->adder-->integrator_out
                integrator_out_port = self.insert_simout(i)
                self.opened_file.writelines('add_line(\'adc_test\','+gain_type+'_port_'+idx+'.Outport(1),{}.Inport(1),\'autorouting\',\'on\');\n'.format(integrator_out_port))
        # gain blocks between vin and other vertices, type as 'b'
        vin_successors = self.opt_dag.successors(0)
        for iv in vin_successors:
            gain_type = 'b'
            eid  = self.opt_dag.get_eid(0,iv)
            if self.opt_dag.es[eid]['type'] != 0:
                gain = self.opt_dag.es['weight'][eid]
                idx  = str(0)+'_'+str(iv)
                self.insert_gainBlock(idx,gain,gain_type)
                # routing, connect: sinwave-->switch-->gain-->adder
                #self.opened_file.writelines('add_line(\'adc_test\',Switch_port.Outport(1),'+gain_type+'_port_'+idx+'.Inport(1),\'autorouting\',\'on\');\n')
                self.opened_file.writelines('add_line(\'adc_test\',SineWave_port.Outport(1),'+gain_type+'_port_'+idx+'.Inport(1),\'autorouting\',\'on\');\n')
                self.opened_file.writelines('add_line(\'adc_test\','+gain_type+'_port_'+idx+'.Outport(1),sum_port_'+str(iv)+'.Inport({}),\'autorouting\',\'on\');\n'.format(self.adder_inputs['sum_port_'+str(iv)]))
                self.adder_inputs['sum_port_'+str(iv)] = self.adder_inputs['sum_port_'+str(iv)] + 1
        # gain blocks between dac and other veryices, type as 'a'
        dac_successors = self.opt_dag.successors(self.num_point-1)
        for dv in dac_successors:
            gain_type = 'a'
            eid  = self.opt_dag.get_eid(self.num_point-1,dv)
            if self.opt_dag.es[eid]['type'] != 0:
                gain = -1*self.opt_dag.es['weight'][eid]
                idx  = str(self.num_point-1)+'_'+str(dv)
                self.insert_gainBlock(idx,gain,gain_type)
                # routing, connect: -->ADC-DAC-->gain-->adder
                self.opened_file.writelines('add_line(\'adc_test\',adda_port.Outport(2),'+gain_type+'_port_'+idx+'.Inport(1),\'autorouting\',\'on\');\n')
                self.opened_file.writelines('add_line(\'adc_test\','+gain_type+'_port_'+idx+'.Outport(1),sum_port_'+str(dv)+'.Inport({}),\'autorouting\',\'on\');\n'.format(self.adder_inputs['sum_port_'+str(dv)]))
                self.adder_inputs['sum_port_'+str(dv)] = self.adder_inputs['sum_port_'+str(dv)] + 1
        # gain blocks between the intermediate node, if forward type as 'gb', if backward type as 'ga'
        for i in range(1,self.order+1):
            m_successors = self.opt_dag.successors(i)
            if i == self.order+1:
                for mv in m_successors:
                    if mv > i+1:
                        gain_type = 'gb'
                        eid  = self.opt_dag.get_eid(i,mv)
                        if self.opt_dag.es[eid]['type'] != 0:
                            gain = self.opt_dag.es['weight'][eid]
                            idx = str(i)+'_'+str(mv)
                            self.insert_gainBlock(idx,gain,gain_type)
                            # routing, connect: -->integrator-->gain-->adder-->
                            self.opened_file.writelines('add_line(\'adc_test\',adda_port.Outport(1),'+gain_type+'_port_'+idx +'.Inport(1),\'autorouting\',\'on\');\n')
                            self.opened_file.writelines('add_line(\'adc_test\','+gain_type+'_port_'+idx +'.Outport(1),sum_port_'+str(mv)+'.Inport({}),\'autorouting\',\'on\');\n'.format(self.adder_inputs['sum_port_'+str(mv)]))
                            self.adder_inputs['sum_port_'+str(mv)] = self.adder_inputs['sum_port_'+str(mv)] + 1
                    elif mv < i-1:
                        gain_type = 'ga'
                        eid  = self.opt_dag.get_eid(i,mv)
                        if self.opt_dag.es[eid]['type'] != 0:
                            gain = -1*self.opt_dag.es['weight'][eid]
                            idx = str(i)+'_'+str(mv)
                            self.insert_gainBlock(idx,gain,gain_type)
                            # routing, connect: -->integrator-->gain-->adder-->
                            self.opened_file.writelines('add_line(\'adc_test\',adda_port.Outport(1),'+gain_type+'_port_'+idx +'.Inport(1),\'autorouting\',\'on\');\n')
                            self.opened_file.writelines('add_line(\'adc_test\','+gain_type+'_port_'+idx +'.Outport(1),sum_port_'+str(mv)+'.Inport({}),\'autorouting\',\'on\');\n'.format(self.adder_inputs['sum_port_'+str(mv)]))
                            self.adder_inputs['sum_port_'+str(mv)] = self.adder_inputs['sum_port_'+str(mv)] + 1
                    else:
                        pass
            else:
                for mv in m_successors:
                    if mv > i+1:
                        gain_type = 'gb'
                        eid  = self.opt_dag.get_eid(i,mv)
                        if self.opt_dag.es[eid]['type'] != 0:
                            gain = self.opt_dag.es['weight'][eid]
                            idx = str(i)+'_'+str(mv)
                            self.insert_gainBlock(idx,gain,gain_type)
                            # routing, connect: -->integrator-->gain-->adder-->
                            self.opened_file.writelines('add_line(\'adc_test\',interg_port_'+str(i)+'.Outport(1),'+gain_type+'_port_'+idx +'.Inport(1),\'autorouting\',\'on\');\n')
                            self.opened_file.writelines('add_line(\'adc_test\','+gain_type+'_port_'+idx +'.Outport(1),sum_port_'+str(mv)+'.Inport({}),\'autorouting\',\'on\');\n'.format(self.adder_inputs['sum_port_'+str(mv)]))
                            self.adder_inputs['sum_port_'+str(mv)] = self.adder_inputs['sum_port_'+str(mv)] + 1
                    elif mv < i-1:
                        gain_type = 'ga'
                        eid  = self.opt_dag.get_eid(i,mv)
                        if self.opt_dag.es[eid]['type'] != 0:
                            gain = -1*self.opt_dag.es['weight'][eid]
                            idx = str(i)+'_'+str(mv)
                            self.insert_gainBlock(idx,gain,gain_type)
                            # routing, connect: -->integrator-->gain-->adder-->
                            self.opened_file.writelines('add_line(\'adc_test\',interg_port_'+str(i)+'.Outport(1),'+gain_type+'_port_'+idx +'.Inport(1),\'autorouting\',\'on\');\n')
                            self.opened_file.writelines('add_line(\'adc_test\','+gain_type+'_port_'+idx +'.Outport(1),sum_port_'+str(mv)+'.Inport({}),\'autorouting\',\'on\');\n'.format(self.adder_inputs['sum_port_'+str(mv)]))
                            self.adder_inputs['sum_port_'+str(mv)] = self.adder_inputs['sum_port_'+str(mv)] + 1
                    else:
                        pass
        self.opened_file.writelines('\n')
    def write_setting(self):
        # write global variables
        self.opened_file.writelines('%%%%%%%%%%%%%%\n')
        self.opened_file.writelines('%%writing the simulation settings\n')
        self.opened_file.writelines('t0=clock;\n')
        self.opened_file.writelines('bw={};\n'.format(self.bw))
        self.opened_file.writelines('Temp={};\n'.format(self.Temp))
        self.opened_file.writelines('k={};\n'.format(self.k))
        self.opened_file.writelines('R={};\n'.format(self.osr))
        self.opened_file.writelines('Fs={};\n'.format(self.Fs))
        self.opened_file.writelines('Ts={};\n'.format(self.Ts))
        self.opened_file.writelines('N={};\n'.format(self.N))
        self.opened_file.writelines('nper={};\n'.format(self.nper))
        self.opened_file.writelines('delta={};\n'.format(self.delta))
        self.opened_file.writelines('finrad={};\n'.format(self.finrad))
        self.opened_file.writelines('alfa={};\n'.format(self.alfa))
        self.opened_file.writelines('Fin={};\n'.format(self.Fin))
        self.opened_file.writelines('Ampl={};\n'.format(self.Ampl))
        self.opened_file.writelines('Ntransient={};\n'.format(self.Ntransient))
        self.opened_file.writelines('NCOMPARATORI={};\n'.format(self.ncomparatori))
        self.opened_file.writelines('Cs={};\n'.format(self.Cs))
        self.opened_file.writelines('Amax={};\n'.format(self.Amax))
        self.opened_file.writelines('sr={};\n'.format(self.sr))
        self.opened_file.writelines('match={};\n'.format(self.match))
        # write simulation settings
        self.opened_file.writelines('options=simset(\'InitialState\', zeros(1,{}), \'RelTol\', 1e-3, \'MaxStep\', 1/Fs);\n'.format(self.order+1))
        self.opened_file.writelines('sim(\'adc_test\', (N+Ntransient)/Fs, options);\n')
        self.opened_file.writelines('w=hann_pv(N);\n')
        self.opened_file.writelines('f=Fin/Fs;\n')
        self.opened_file.writelines('fB=N*(bw/Fs);\n')
        self.opened_file.writelines('yy1=zeros(1,N);\n')
        self.opened_file.writelines('yy1=yout(2+Ntransient:1+N+Ntransient)\';\n')
        self.opened_file.writelines('ptot=zeros(1,N);\n')
        self.opened_file.writelines('[snr,ptot]=calcSNR(yy1(1:N),f,1,fB,w,N);\n')
        self.opened_file.writelines('Rbit=(snr-1.76)/6.02;\n')
        for i in range(1,self.order+1):
            self.opened_file.writelines('integrator_constr_{}=max(abs(integrators_out_{}));\n'.format(i,i))
        # save results
        self.opened_file.writelines('fileID = fopen(\'sim_res.txt\',\'w\');\n')
        self.opened_file.writelines('fprintf(fileID,\'%1.3f %1.3f %1.3f %1.3f %1.3f %1.3f\\n\',snr,Rbit,integrator_constr_{},integrator_constr_{},integrator_constr_{},integrator_constr_{});\n'.format(1,2,3,4))
        self.opened_file.writelines('fclose(fileID);\n')
        # close and save model
        self.opened_file.writelines('close_system(\'adc_test\',1);\n')

    def cal_power(self,SNDR):
        # only consider the power consumption of the integrator
        # the power consumption of the quantizer will be considered in subsequent versions

        p_singal = ((1.8*2)**2)/8 # (vdd*2)^2/8
        p_noise  = 10**(np.log10(p_singal)-SNDR/10)
        c_eq     = 2*self.k*self.Temp/(p_noise*self.osr)

        k1_eid   = self.opt_dag.get_eid(0,1)
        k1       = self.opt_dag.es['weight'][k1_eid]
        c_eq     = c_eq/k1

        # empirically, the gbw of the first-stage integrator is used to calculate power
        gbw_tmp  = self.opt_dag.vs[1]['gbw']* self.Fs
        gm    = gbw_tmp*c_eq*2*np.pi
        id    = gm/15
        power = id*2*1.8

        return power * 1e6 # pW (uW*1e6)

    def cal_fom(self,SNDR,POWER):
        POWER = POWER/1e6 # switch to uW
        fom = SNDR + 10*np.log10(self.bw/POWER) # >185 will be excellent
        return fom

    def run(self,matlab_eng):
        # gen executable file
        self.gen_mdoel()
        self.write_setting()
        self.opened_file.close()
        # run
        matlab_eng.adc_gen_sim(nargout=0)
        '''
        eng = matlab.engine.start_matlab()
        eng.adc_gen_sim(nargout=0)
        eng.exit()
        '''
        # get simulation results
        sim =  os.path.exists('./sim_res.txt')
        if sim==True:
            res  = open('./sim_res.txt','r')
            line = res.readlines()[0]
            stripped = line.strip()
            match = re.search(r'(\S+)\s(\S+)\s(\S+)\s(\S+)\s(\S+)\s(\S+)$',stripped)
            if match:
                self.sndr   = float(match.groups()[0])
                self.enob   = float(match.groups()[1])
                integrator_constr_1 = float(match.groups()[2])
                integrator_constr_2 = float(match.groups()[3])
                integrator_constr_3 = float(match.groups()[4])
                integrator_constr_4 = float(match.groups()[5])
            res.close()
        # return
        sndr   = self.sndr
        enob   = self.enob
        power  = self.cal_power(sndr)
        fom    = self.cal_fom(sndr,power)
        return sndr, enob, power, fom, integrator_constr_1, integrator_constr_2, integrator_constr_3, integrator_constr_4


