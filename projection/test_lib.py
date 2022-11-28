import warnings

warnings.filterwarnings('ignore')

import utils
import fw_lib
import irbp_lib
import numpy as np
from numpy import linalg as LA

if __name__ == '__main__':
    # np.random.seed(4)
    # random.seed(4)
    
    dataSize = int(1e5) # the size of the input vector

    #======== parameter for Lp-ball constraint =========
    p = 0.8
    #========
    average_num = 3

    # Storing Results 
    list_alpha = np.zeros((2, average_num))
    list_beta = np.zeros((2, average_num))    
    list_obj = np.zeros((2, average_num))
    list_time = np.zeros((2, average_num))
    list_nonzero = np.zeros((2, average_num))
    

    for k in range(average_num):

        y = utils.point_projected(dataSize) # standard normal distribution
        radius = 0.01 * LA.norm(y, p) ** p

        """
         Method I: The benchmark algorithm: IRBP
        """
        
        #%% Initialization
        
        x_ini = 0.3 *  (radius ** (1/p)) * (np.abs(y,dtype=np.float64) / LA.norm(y,p))
        epsilon = 0.6 * (radius ** (1/p)) * (np.abs(y,dtype=np.float64) / LA.norm(y,p))

        x_irbp, dual, t_elapsed_irbp, count = irbp_lib.get_lp_ball_projection(x_ini, y, p, radius, epsilon)  # type: ignore

        # Check the location of nonzero elements
        print('-' * 40, 'IBRP method', '-' * 40)
        act_ind = np.nonzero(x_irbp)[0]
        lamb = sum(y[act_ind] - x_irbp[act_ind]) / sum(p * np.sign(x_irbp[act_ind]) * abs(x_irbp[act_ind]) ** (p - 1))
        alpha_res_irbp = (1. / dataSize) * LA.norm((x_irbp - y) * x_irbp + lamb * p * (abs(x_irbp) ** p), 1)  # type: ignore
        beta_res_irbp =  abs(LA.norm(x_irbp, p) ** p - radius)
        
        print(f"k={k}, IterNum={count}, Obj={utils.least_square_loss(x_irbp, y)}, alpha_res={alpha_res_irbp}, beta={beta_res_irbp}, #nonzero={len(np.nonzero(x_irbp)[0])}, time={t_elapsed_irbp}")# type: ignore
        print('-' * 100)
        

        '''
        Method II: the proposed hybrid Frank-Wolfe algorithm for solving the projection problem
        '''
        # x_fw, list_obj_fw, list_fea_fw, t_elapsed_fw, iter_, iter_proj, times = FW_Proj.Frank_Wolfe_Lp(x_ini, y, p, radius)
        x_fw, t_elapsed_fw, iter_ = fw_lib.Frank_Wolfe_Lp(x_ini, y, p, radius)
        
        
        print('-' * 40, 'Frank-Wolfe method', '-' * 40)
        act_ind = np.nonzero(x_fw)[0]
        lamb_fw = sum(y[act_ind] - x_fw[act_ind]) / sum(p * np.sign(x_fw[act_ind]) * abs(x_fw[act_ind]) ** (p - 1))  # type: ignore
        alpha_res = (1. / dataSize) * LA.norm((x_fw - y) * x_fw + lamb_fw * p * (abs(x_fw) ** p), 1)
        beta_res = abs(LA.norm(x_fw, p) ** p - radius)
        print(f"k={k}, IterNum={iter_}, Obj={utils.least_square_loss(x_fw, y)}, alpha_res={alpha_res}, beta={beta_res}, #nonzero={len(np.nonzero(x_fw)[0])}, time={t_elapsed_fw}")
        print('-' * 100)


        list_alpha[0][k] = alpha_res
        list_beta[0][k] = beta_res
        list_obj[0][k] = utils.least_square_loss(x_fw, y)
        list_nonzero[0][k] = len(np.nonzero(x_fw)[0])
        list_time[0][k] = t_elapsed_fw
        
        list_alpha[1][k] = alpha_res_irbp
        list_beta[1][k] = beta_res_irbp
        list_obj[1][k] = utils.least_square_loss(x_irbp, y)
        list_nonzero[1][k] = len(np.nonzero(x_irbp)[0])
        list_time[1][k] = t_elapsed_irbp

        
        
        
    # np.savez('data_arr08_5.npz', x = list_alpha, y = list_beta, z = list_obj, a = list_nonzero, b = list_time)
    
    
        #%% Store the 20 times results. 
        # np.save('list_alpha.npy', list_alpha)
        # np.save('list_beta.npy', list_beta)
        # np.save('list_obj.npy', list_obj)
        # np.save('list_nonzero.npy', list_nonzero)
        # np.save('list_time.npy', list_time)
        
        #%% Data processing
        
    # Time_matrix = np.vstack(Time_seq)
    # Obj_matrix = np.vstack(Obj_seq)
    # Fea_matrix = np.vstack(Fea_seq)
    # plt.figure()
    # plt.plot(times, list_obj_fw)
    # plt.show()
    
        