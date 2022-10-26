from matplotlib import pyplot as plt
import math
from pykalman import KalmanFilter
import numpy as np
import pandas as pd


def filtered_kalman(values):
    kf = KalmanFilter(transition_matrices=np.array([[1, 1], [0, 1]]),
                      transition_covariance=0.0001 * np.eye(2)) # np.eyeは単位行列
    smoothed = kf.em(values).smooth(values)[0]
    filtered = kf.em(values).filter(values)[0]
    return smoothed, filtered


if __name__ == '__main__':
    key = 'L凸90deg'
    w = 'z'
    df = pd.read_csv('input/' + key + '.csv')
    x = df['Time (s)']
    a = df['Linear Acceleration z (m/s^2)']

    smoothed, filtered = filtered_kalman(a)
    a_kal = smoothed[:, 0]

    v_kal = [0] * len(a_kal)
    #速度の算出
    for i in range(len(v_kal) - 1):
        v_kal[i+1] = (a_kal[i] + a_kal[i+1]) * (x[i+1] - x[i]) / 2 + v_kal[i]

    df_out = pd.DataFrame()
    df_out['t'] = x
    df_out['a'] = a
    df_out['a_kal'] = a_kal
    df_out['v_kal'] = v_kal
    df_out.to_csv('out_v' + w + '_' + key + '.csv')

    v_min = min(v_kal)
    print('Assumed velocity is ' + str(v_min) + ' [m/s]')


    plt.figure(figsize=(16, 9), dpi=80)
    plt.title(key)
    plt.plot(x, a, label='Raw Data', color = 'k')
    plt.plot(x, a_kal, label='Kalman Smoothed', color = 'r')
    plt.xlabel('time [s]')
    plt.ylabel('z-acceleration [m/s^2]')
    plt.legend()
    plt.show()

    plt.plot(x, v_kal, color = 'b')
    plt.title(key)
    plt.xlabel('time [s]')
    plt.ylabel('z-velocity [m/s]')
    plt.grid()
    plt.show()
