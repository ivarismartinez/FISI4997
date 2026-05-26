#!/usr/bin/python3.11
#
#python version 3.7 is needed to get scipy version > 0.18 (which supports bounds).
#Anish May 24, 2020
#
#Included fitting of Voigt profile. 
#Change the class name to GAUSSVOIGT
#Anish April 14, 2021

#An example of user supplied function to get input data
#def getdat(): 
#   lab2='G34.20+0.0';
#   f=open("../paper/figs/G34P2.TXT", "r")
#   lines=f.readlines()
#   f.close()
#   x = np.array([np.float(line.split()[1]) for line in lines])
#   x = x/1000.0
#   y = np.array([np.float(line.split()[2]) for line in lines])
#   wt=np.ones(y.shape)
#   return x,y,wt


import numpy as np
import sys
import scipy.io as spio
import matplotlib.pyplot as plt
import matplotlib.rcsetup
import matplotlib
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from scipy.optimize import curve_fit
from scipy.special import wofz

sigma2fwhm=2.0*(np.sqrt(2*np.log(2)))

class GAUSSVOIGTFIT :
    def __init__ ( self, verbose=0 ) :

       self.verbose=verbose
       #font = {'family':'serif', 'serif':'Computer Modern', 'weight' : 'bold'}
       font = {'family':'DejaVu Sans',  'weight' : 'bold', 'size' : 5}
       #dpi = 300
       #matplotlib.rc('font', **font)
       #matplotlib.rc('text', usetex='true')
       #matplotlib.rc('figure', dpi= dpi)
       # These are the "Tableau 20" colors as RGB.
       self.tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
                    (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
                    (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
                    (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
                    (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]
       
       # Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.
       for i in range(len(self.tableau20)):
           r, g, b = self.tableau20[i]
           self.tableau20[i] = (r / 255., g / 255., b / 255.)
       
       self.fig = plt.figure(figsize = (8,7)) #(4,3))
       self.ax0 = self.fig.add_axes([0.15, 0.15, 0.7, 0.7])
       self.ax0.grid(True, linestyle='--', alpha=0.5)
       self.ax0.grid(True, which='minor', linestyle=':', alpha=0.3)
       return

    #Voigt function
    def voigt(self, x, *p):
       y = np.zeros(x.size)
       for i in range(int(len(p)/4.0)):
         amp=p[0+i*4]
         center = p[1+i*4]
         gamma = p[2+i*4] 
         sigma = p[3+i*4]
         v = np.real(wofz(((x-center) + 1j*gamma)/sigma/np.sqrt(2))) / sigma\
                                                           /np.sqrt(2*np.pi)

         v = v/np.max(np.max(v))*amp
         y = y + v 
       return y

    #Sum of multiple gaussian. The parameters are amp, sigma, cen of the Gaussian
    def gaussian(self, x, *p):
       y = np.zeros(x.size)
       for i in range(int(len(p)/3.0)):
         y = y + p[0+i*3]*np.exp((-(x-p[1+i*3])**2)/(2.0*p[2+i*3]**2))
       return y
    
    #Initialize voigt fit; set values and constraints for the variables, 
    def initvfit(self, x,y,wt):
       self.ax0.cla()
       self.ax0.plot(x, y, linewidth = 2, color=self.tableau20[1])
       self.ax0.grid(True, linestyle='--', alpha=0.5)
       self.ax0.grid(True, which='minor', linestyle=':', alpha=0.3)
       self.fig.canvas.draw()
    
       print('Select the X-range for Voigt fit')
       xmin, xmax=plt.xlim()
       
       pltval = plt.ginput(2,0,True)
       xlimit = [pltval[0][0], pltval[1][0]]
       xlimit.sort()
       if(xlimit[0] < xmin):
         xlimit[0] = xmin
       
       if(xlimit[1] > xmax):
         xlimit[1] = xmax;
       
       x1 = x[np.logical_and(x>xlimit[0], x<xlimit[1])];
       y1 = y[np.logical_and(x>xlimit[0], x<xlimit[1])];
       wt1 = wt[np.logical_and(x>xlimit[0], x<xlimit[1])];
       
       self.ax0.cla()
       self.ax0.grid(True, linestyle='--', alpha=0.5)
       self.ax0.grid(True, which='minor', linestyle=':', alpha=0.3)
       self.ax0.plot(x1, y1, linewidth = 2, color=self.tableau20[1])
       self.fig.canvas.draw()
       
       nvoigt = int(input("Enter the number of Voigt profiles : "))

       init_vals=np.zeros(nvoigt*4)
       lbound=np.ones(init_vals.shape)*(-np.inf)
       ubound=np.ones(init_vals.shape)*np.inf
       
       print("Enter the initial guess for the parameters of the Voigt")
       for m in range(nvoigt):
         print("Voigt number : {0:3d} ".format(m+1))
    
         print("Enter top and bottom points of Voigt amplitude ")
         pltval = plt.ginput(2,0,True);
         init_vals[0+m*4] = pltval[0][1] - pltval[1][1]
         print("{0:4.1f}".format(init_vals[0+m*4]))
    
         fix=-1
         while fix < 0 or fix > 4:
            fix = int(input("Vary this parameter ? (vary/fix/upper/lower/upper-lower - 0/1/2/3/4) : "))
    
         if fix == 1:
             amp = float(input("Enter the amplitude of the Voigt : "))
             init_vals[0+m*4] = amp
             print("{0:4.1f}".format(init_vals[0+m*4]))
             lbound[0+m*4] = amp-np.abs(amp)*0.001
             ubound[0+m*4] = amp+np.abs(amp)*0.001
         elif fix == 2:
             amp = float(input("Enter the upper bound for the amplitude of the Voigt : "))
             ubound[0+m*4] = amp
         elif fix == 3:
             amp = float(input("Enter the lower bound for the amplitude of the Voigt : "))
             lbound[0+m*4] = amp
         elif fix == 4:
             amp = float(input("Enter the upper bound for the amplitude of the Voigt : "))
             ubound[0+m*4] = amp
             amp = float(input("Enter the lower bound for the amplitude of the Voigt : "))
             lbound[0+m*4] = amp
       
         print("Enter center of the Voigt : ")
         pltval = plt.ginput(1,0,True)
         init_vals[1+m*4] = pltval[0][0]
         print("{0:5.1f}".format(init_vals[1+m*4]))
    
         fix=-1
         while fix < 0 or fix > 4:
            fix = int(input("Vary this parameter ? (vary/fix/upper/lower/upper-lower - 0/1/2/3/4) : "))
    
         if fix == 1:
             cen = float(input("Enter the center of the Voigt : "))
             init_vals[1+m*4] = cen
             print("{0:5.1f}".format(init_vals[1+m*4]))
             lbound[1+m*4] = cen-np.abs(cen)*0.001
             ubound[1+m*4] = cen+np.abs(cen)*0.001
         elif fix == 2:
             cen = float(input("Enter the upper bound for the center of the Voigt : "))
             ubound[1+m*4] = cen
         elif fix == 3:
             cen = float(input("Enter the lower bound for the center of the Voigt : "))
             lbound[1+m*4] = cen
         elif fix == 4:
             cen = float(input("Enter the upper bound for the center of the Voigt : "))
             ubound[1+m*4] = cen
             cen = float(input("Enter the lower bound for the center of the Voigt : "))
             lbound[1+m*4] = cen
       
         print("Enter FWHM width of the Voigt : ")
         pltval = plt.ginput(2,0,True);
         init_vals[2+m*4] = np.abs(pltval[0][0] - pltval[1][0])
         print("{0:3.1f}".format(init_vals[2+m*4]))
    
         fix=-1
         while fix < 0 or fix > 4:
            fix = int(input("Vary this parameter ? (vary/fix/upper/lower/upper-lower - 0/1/2/3/4) : "))
    
         if fix == 1:
             width = float(input("Enter the FWHM width of the Voigt : "))
             init_vals[2+m*4] = width 
             print("{0:3.1f}".format(init_vals[2+m*4]))
             lbound[2+m*4] = width-np.abs(width)*0.001 
             ubound[2+m*4] = width+np.abs(width)*0.001
         elif fix == 2:
             width = float(input("Enter the upper bound for the FWHM width of the Voigt : "))
             ubound[2+m*4] = width
         elif fix == 3:
             width = float(input("Enter the lower bound for the FWHM width of the Voigt : "))
             lbound[2+m*4] = width
         elif fix == 4:
             width = float(input("Enter the upper bound for the FWHM width of the Voigt : "))
             ubound[2+m*4] = width
             width = float(input("Enter the lower bound for the FWHM width of the Voigt : "))
             lbound[2+m*4] = width


         print(" ")
         fix=-1
         while fix < 0 or fix > 4:
            fix = int(input("Vary FWHM Doppler width parameter ? (vary/fix/upper/lower/upper-lower - 0/1/2/3/4) : "))
    
         if fix == 0:
             dwidth = float(input("Enter the FWHM Doppler width (in units of x-axis): "))
             init_vals[3+m*4]=dwidth/sigma2fwhm #To sigma
         elif fix == 1:
             dwidth = float(input("Enter the FWHM Doppler width (in units of x-axis): "))
             dwidth = dwidth/sigma2fwhm
             init_vals[3+m*4]=dwidth #To sigma
             print("{0:3.1f}".format(init_vals[3+m*4]))
             lbound[3+m*4] = dwidth-np.abs(dwidth)*0.001 
             ubound[3+m*4] = dwidth+np.abs(dwidth)*0.001
         elif fix == 2:
             dwidth = float(input("Enter the FWHM Doppler width (in units of x-axis): "))
             init_vals[3+m*4]=dwidth/sigma2fwhm #To sigma
             dwidth = float(input("Enter the upper bound for the FWHM Doppler width (in units of x-axis) : "))
             ubound[3+m*4] = dwidth/sigma2fwhm #To sigma
         elif fix == 3:
             dwidth = float(input("Enter the FWHM Doppler width (in units of x-axis): "))
             init_vals[3+m*4]=dwidth/sigma2fwhm #To sigma
             dwidth = float(input("Enter the lower bound for the FWHM Doppler width (in units of x-axis) : "))
             lbound[3+m*4] = dwidth/sigma2fwhm #To sigma
         elif fix == 4:
             dwidth = float(input("Enter the FWHM Doppler width (in units of x-axis): "))
             init_vals[3+m*4]=dwidth/sigma2fwhm #To sigma
             dwidth = float(input("Enter the upper bound for the FWHM Doppler width (in units of x-aixs) : "))
             ubound[3+m*4] = dwidth/sigma2fwhm #To sigma
             dwidth = float(input("Enter the lower bound for the FWHM Doppler width (in units of x-axis) : "))
             lbound[3+m*4] = dwidth/sigma2fwhm #To sigma

         print(ubound, lbound)
         #Convert Voigt width to gamma using Eq 1 of Payne et al 1994 
         width=init_vals[2+m*4] #This is in FWHM
         dwidth=init_vals[3+m*4]*sigma2fwhm #to FWHM
         init_vals[2+m*4]=8.83*width/2-np.sqrt(61.39*(width/2)**2+16.67*(dwidth/2)**2) #this is HWHM or gamma 
         if(init_vals[2+m*4] < 0):
            print("Gamma is negative !!!")
            sys.exit(1) 
         ubound[2+m*4]=ubound[2+m*4]/2 #this is HWHM or gamma (approximate bound)
         lbound[2+m*4]=lbound[2+m*4]/2 #this is HWHM or gamma (approximate bound)
         
       return x1,y1,wt1,init_vals,(lbound,ubound) 

    #Initialize gaussian fit; set values and constraints for the variables, 
    def initgfit(self, x,y,wt,hdr):
       self.ax0.cla()
       self.ax0.plot(x, y, linewidth = 2, color=self.tableau20[1])
       self.ax0.grid(True, linestyle='--', alpha=0.5)
       self.ax0.grid(True, which='minor', linestyle=':', alpha=0.3)
       self.ax0.set_title(hdr)
       self.fig.canvas.draw()
    
       print('Select the X-range for Gaussian fit')
       xmin, xmax=plt.xlim()
       
       pltval = plt.ginput(2,0,True)
       xlimit = [pltval[0][0], pltval[1][0]]
       xlimit.sort()
       if(xlimit[0] < xmin):
         xlimit[0] = xmin
       
       if(xlimit[1] > xmax):
         xlimit[1] = xmax;
       
       x1 = x[np.logical_and(x>xlimit[0], x<xlimit[1])];
       y1 = y[np.logical_and(x>xlimit[0], x<xlimit[1])];
       wt1 = wt[np.logical_and(x>xlimit[0], x<xlimit[1])];
       
       self.ax0.cla()
       self.ax0.grid(True, linestyle='--', alpha=0.5)
       self.ax0.grid(True, which='minor', linestyle=':', alpha=0.3)
       self.ax0.plot(x1, y1, linewidth = 2, color=self.tableau20[1])
       self.ax0.set_title(hdr)
       self.fig.canvas.draw()
       
       ngauss = int(input("Enter the number of Gaussians : "))
       init_vals=np.zeros(ngauss*3)
       lbound=np.ones(init_vals.shape)*(-np.inf)
       ubound=np.ones(init_vals.shape)*np.inf
       
       print("Enter the initial guess for the parameters of the Gaussians")
       for m in range(ngauss):
         print("Gaussian number : {0:3d} ".format(m+1))
    
         print("Enter top and bottom points of Gaussian amplitude ")
         pltval = plt.ginput(2,0,True);
         init_vals[0+m*3] = pltval[0][1] - pltval[1][1]
         print("{0:4.1f}".format(init_vals[0+m*3]))
    
         fix=-1
         while fix < 0 or fix > 4:
            fix = int(input("Vary this parameter ? (vary/fix/upper/lower/upper-lower - 0/1/2/3/4) : "))
    
         if fix == 1:
             amp = float(input("Enter the amplitude of the Gaussian : "))
             init_vals[0+m*3] = amp
             print("{0:4.1f}".format(init_vals[0+m*3]))
             lbound[0+m*3] = amp-np.abs(amp)*0.001
             ubound[0+m*3] = amp+np.abs(amp)*0.001
         elif fix == 2:
             amp = float(input("Enter the upper bound for the amplitude of the Gaussian : "))
             ubound[0+m*3] = amp
         elif fix == 3:
             amp = float(input("Enter the lower bound for the amplitude of the Gaussian : "))
             lbound[0+m*3] = amp
         elif fix == 4:
             amp = float(input("Enter the upper bound for the amplitude of the Gaussian : "))
             ubound[0+m*3] = amp
             amp = float(input("Enter the lower bound for the amplitude of the Gaussian : "))
             lbound[0+m*3] = amp
       
         print("Enter center of the Gaussian : ")
         pltval = plt.ginput(1,0,True)
         init_vals[1+m*3] = pltval[0][0]
         print("{0:5.1f}".format(init_vals[1+m*3]))
    
         fix=-1
         while fix < 0 or fix > 4:
            fix = int(input("Vary this parameter ? (vary/fix/upper/lower/upper-lower - 0/1/2/3/4) : "))
    
         if fix == 1:
             cen = float(input("Enter the center of the Gaussian : "))
             init_vals[1+m*3] = cen
             print("{0:5.1f}".format(init_vals[1+m*3]))
             lbound[1+m*3] = cen-np.abs(cen)*0.001
             ubound[1+m*3] = cen+np.abs(cen)*0.001
         elif fix == 2:
             cen = float(input("Enter the upper bound for the center of the Gaussian : "))
             ubound[1+m*3] = cen
         elif fix == 3:
             cen = float(input("Enter the lower bound for the center of the Gaussian : "))
             lbound[1+m*3] = cen
         elif fix == 4:
             cen = float(input("Enter the upper bound for the center of the Gaussian : "))
             ubound[1+m*3] = cen
             cen = float(input("Enter the lower bound for the center of the Gaussian : "))
             lbound[1+m*3] = cen
       
         print("Enter FWHM width of the Gaussian : ")
         pltval = plt.ginput(2,0,True);
         init_vals[2+m*3] = np.abs(pltval[0][0] - pltval[1][0])
         init_vals[2+m*3] = init_vals[2+m*3]/sigma2fwhm; #To sigma
         print("{0:3.1f}".format(init_vals[2+m*3]))
    
         fix=-1
         while fix < 0 or fix > 4:
            fix = int(input("Vary this parameter ? (vary/fix/upper/lower/upper-lower - 0/1/2/3/4) : "))
    
         if fix == 1:
             width = float(input("Enter the FWHM width of the Gaussian : "))
             init_vals[2+m*3] = width/sigma2fwhm 
             print("{0:3.1f}".format(init_vals[2+m*3]))
             lbound[2+m*3] = width/sigma2fwhm-np.abs(width/sigma2fwhm)*0.001 
             ubound[2+m*3] = width/sigma2fwhm+np.abs(width/sigma2fwhm)*0.001
         elif fix == 2:
             width = float(input("Enter the upper bound for the FWHM width of the Gaussian : "))
             ubound[2+m*3] = width/2.0/(np.sqrt(2*np.log(2)))
         elif fix == 3:
             width = float(input("Enter the lower bound for the FWHM width of the Gaussian : "))
             lbound[2+m*3] = width/2.0/(np.sqrt(2*np.log(2)))
         elif fix == 4:
             width = float(input("Enter the upper bound for the FWHM width of the Gaussian : "))
             ubound[2+m*3] = width/2.0/(np.sqrt(2*np.log(2)))
             width = float(input("Enter the lower bound for the FWHM width of the Gaussian : "))
             lbound[2+m*3] = width/2.0/(np.sqrt(2*np.log(2)))
    
       return x1,y1,wt1,init_vals,(lbound,ubound) 

    def prnvoigtvals(self, p=None, err=None, lbound=None, ubound=None, covar=None, verbose=0):
       if err is not None:
          out=open('voigtfit.out', 'w')
          print("Writing fitted values to voigtfit.out")

       if err is None and verbose:
          outstr="Amplitude     Center   Lwidth(FWHM)  Dwidth(FWHM)" 
          print(outstr) 
          for i in range(int(len(p)/4.0)):
            outstr="{0:7.2f}, {1:7.2f}, {2:7.2f}, {3:7.2f}".format(p[0+i*4], p[1+i*4], p[2+i*4]*2, p[3+i*4]*sigma2fwhm)
            print(outstr)
       elif err is not None:
          outstr="Amplitude (err)   Center(err)   LFWHM (err) DFWHM (err)" 
          if verbose:
             print(outstr) 
          out.write(outstr+'\n')
          padding = " "
          for i in range(int(len(p)/4.0)):
            outstr="{0:7.2f}({ 1:7.2f}), {2:7.2f}({3:7.2f}), {4:7.2f}({5:7.2f}), {6:7.2f}({7:7.2f})".format(p[0+i*4],err[0+i*4],p[1+i*4],err[1+i*4],
                    p[2+i*4]*2,err[2+i*4]*2, p[3+i*4]*sigma2fwhm,err[3+i*4]*sigma2fwhm)
            if verbose:
               print(outstr)
            out.write(outstr+'\n')
          out.close()

       if lbound is not None and verbose:
          print("Lower bounds")
          for i in range(int(len(lbound)/4.0)):
            print("{0:6.1f}, {1:6.1f}, {2:6.1f}, {3:6.1f}".format(lbound[0+i*4], lbound[1+i*4], lbound[2+i*4]*2, lbound[3+i*4]*sigma2fwhm))

       if ubound is not None and verbose:
          print("Upper bounds")
          for i in range(int(len(ubound)/4.0)):
            print("{0:6.1f}, {1:6.1f}, {2:6.1f}, {3:6.1f}".format(ubound[0+i*4], ubound[1+i*4], ubound[2+i*4]*2, ubound[3+i*4]*sigma2fwhm))

       if covar is not None and verbose:
          print("Covariance matrix")
          print('\n'.join([''.join(['{:7.2f}'.format(item) for item in row]) for row in covar]))
       
    def prnvals(self, p=None, err=None, lbound=None, ubound=None, covar=None, verbose=0, outfile=None):
       if err is not None:
          out=open(outfile, 'w')
          print("Writing fitted values to "+outfile)

       if err is None and verbose:
          outstr="Amplitude     Center   FWHM" 
          print(outstr) 
          for i in range(int(len(p)/3.0)):
            outstr="{0:9.6f}, {1:9.6f}, {2:7.2f}".format(p[0+i*3], p[1+i*3], p[2+i*3]*sigma2fwhm)
            print(outstr) 
       elif err is not None:
          outstr="Amplitude (err)   Center(err)   FWHM (err)" 
          if verbose:
             print(outstr) 
          out.write(outstr+'\n')
          for i in range(int(len(p)/3.0)): 
            outstr="{0:9.6f}({1:9.6f}), {2:7.2f}({3:7.2f}), {4:7.2f}({5:7.2f})".format(p[0+i*3],err[0+i*3],p[1+i*3],err[1+i*3],p[2+i*3]*sigma2fwhm,err[2+i*3]*sigma2fwhm)
            if verbose:
               print(outstr)
            out.write(outstr+'\n')
          out.close()

       if lbound is not None and verbose:
          print("Lower bounds")
          for i in range(int(len(lbound)/3.0)):
            print("{0:6.1f}, {1:6.1f}, {2:6.1f}".format(lbound[0+i*3], lbound[1+i*3], lbound[2+i*3]*sigma2fwhm))

       if ubound is not None and verbose:
          print("Upper bounds")
          for i in range(int(len(ubound)/3.0)):
            print("{0:6.1f}, {1:6.1f}, {2:6.1f}".format(ubound[0+i*3], ubound[1+i*3], ubound[2+i*3]*sigma2fwhm))

       if covar is not None and verbose:
          print("Covariance matrix")
          print('\n'.join([''.join(['{:7.2f}'.format(item) for item in row]) for row in covar]))

    def gaussfit(self, x1,y1,wt1,hdr,outfile):
       self.ax0.plot(x1, y1, linewidth = 2, color=self.tableau20[1])
       self.ax0.set_title(hdr)
       self.fig.canvas.draw()
       while 1==1:
          x,y,wt,init_vals,bounds=self.initgfit(x1,y1,wt1,hdr)
          if self.verbose:
             print("Initial values and bounds")
             self.prnvals(init_vals, None, bounds[0], bounds[1], verbose=self.verbose) 
#          best_vals, covar = curve_fit(self.gaussian, x, y, p0=init_vals, sigma=np.sqrt(wt),bounds=bounds)
          best_vals, covar = curve_fit(self.gaussian, x, y, p0=init_vals, sigma=np.sqrt(wt))
          err=np.sqrt(covar.diagonal())
          self.prnvals(best_vals, err, covar=covar, verbose=self.verbose,outfile=outfile) 
          self.ax0.plot(x,self.gaussian(x, *best_vals))
          self.ax0.set_title(hdr)
          self.fig.canvas.draw()
          plt.pause(0.1)
          reply = input("Is the gauss fit ok ? Y/N [N]: ")
          if reply.lower() == 'y':
             sys.exit(0)
          
       plt.show()

    def voigtfit(self, x1,y1,wt1):
       self.ax0.plot(x1, y1, linewidth = 2, color=self.tableau20[1])
       self.fig.canvas.draw()
       while 1==1:
          x,y,wt,init_vals,bounds=self.initvfit(x1,y1,wt1)
          if self.verbose:
             print("Initial values and bounds")
             self.prnvoigtvals(init_vals, None, bounds[0], bounds[1], verbose=self.verbose) 
          best_vals, covar = curve_fit(self.voigt, x, y, p0=init_vals, sigma=np.sqrt(wt),bounds=bounds)
          err=np.sqrt(covar.diagonal())
          self.prnvoigtvals(best_vals, err, covar=covar, verbose=self.verbose) 
          self.ax0.plot(x,self.voigt(x, *best_vals))
          self.fig.canvas.draw()
          reply = input("Is the voigtfit fit ok ? Y/N [N]: ")
          if reply.lower() == 'y':
             sys.exit(0)
          
       plt.show()


