#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 12:11:52 2018

@author: raphaedon
"""

import numpy as np
import matplotlib.pyplot as plt

def ce():
    T = 1
    np.random.seed(20180612)
    px = np.random.uniform(size=10000)
    x = -T * np.log(px)
    
    x1 = np.linspace(0, 10, 10000)
    y1 = np.exp(-x1/T)
    n, bins, patches = (plt.hist(x, bins=10, density=True, orientation='horizontal', 
                                 alpha=0.75, edgecolor='black', linewidth=1.))
    plt.plot(y1, x1, 'r--', label=r'$P\propto e^{-E/T}$')
    plt.xlabel(r'$\mathrm{Prob(Energy\ level\ =\ x)}$')
    plt.ylabel(r'$\mathrm{Energy\ level:\ x}$')
    plt.title(r'$\mathrm{Histogram\ of\ Canonical\ Ensemble:}\ T=1$')
    plt.grid(True)
    plt.axis([0, 1.0, 0, 10])
    plt.legend()
    plt.show()
    
    T = 2
    x = - T * np.log(px)
    y2 = np.exp(-x1/T)/T
    n, bins, patches = (plt.hist(x, bins=20, density=True, orientation='horizontal', 
                                 alpha=0.75, edgecolor='black', linewidth=1.))
    plt.plot(y2, x1, 'g--', label=r'$P\propto e^{-E/T}$')
    plt.xlabel(r'$\mathrm{Prob(Energy\ level\ =\ x)}$')
    plt.ylabel(r'$\mathrm{Energy\ level:\ x}$')
    plt.title(r'$\mathrm{Histogram\ of\ Canonical\ Ensemble:}\ T=2$')
    plt.grid(True)
    plt.axis([0, 1.0, 0, 10])
    plt.legend()
    plt.show()
    
def gaussian():
    x = np.linspace(0, 4)
    y = np.exp(-(x-2)**2/(2*0.5))/np.sqrt(2*np.pi*0.5)
    plt.plot(x, y, 'r--', label=r'$p(\theta|x)=\frac{1}{\sqrt{2\pi\delta^2}}e^{-\frac{(x-\mu)^2}{2\delta^2}}$')
    plt.xlabel(r'$\mathrm{parameter:\ \theta}$')
    plt.ylabel(r'$\mathrm{p(\theta|x)}$')
    plt.title(r'$\mathrm{Gaussian\ Distribution:}\ \mu=2,\ \delta^2=0.01$')
    plt.grid(True)
    plt.legend()
    plt.show()
    
    x = np.linspace(0, 4)
    y = 1 - np.exp(-(x-2)**2/(2*0.5))/np.sqrt(2*np.pi*0.5)
    plt.plot(x, y, 'g--', label=r'$p(\theta|x)=1-\frac{1}{\sqrt{2\pi\delta^2}}e^{-\frac{(x-\mu)^2}{2\delta^2}}$')
    plt.xlabel(r'$\mathrm{\theta}$')
    plt.ylabel(r'$\mathrm{p(\theta|x)}$')
    plt.title(r'$\mathrm{Inverse\ Gaussian\ Distribution:}\ \mu=2,\ \delta^2=0.01$')
    plt.grid(True)
    plt.legend()
    plt.show()

def main():
    gaussian()
    
    
if __name__ == '__main__':
    main()