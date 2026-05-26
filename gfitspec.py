# !/usr/bin/python3.11
#

import numpy as np
import sys
from gaussvoigt import GAUSSVOIGTFIT 

g=GAUSSVOIGTFIT()

def getdat(filename):

   f = open(filename, "r")
   lines = f.readlines()
   f.close()

   datalines = []
   for line in lines:
      line = line.strip()

      # Skip comment lines
      if line.startswith('%'):
         continue
      
      datalines.append(line)

   hdr = datalines[0]
   x = np.array([float(line.split()[0]) for line in datalines[1:]])
   y = np.array([float(line.split()[1]) for line in datalines[1:]])
   wt = np.ones(y.shape)
   wt[np.isnan(y)] = 0.0
   return x,y,wt,hdr

filenames=np.array([
                  #   '/Users/ivarismartinez/Desktop/Astro_Project_2/spectrum_l=277_b=-1.txt', # done
                  #   '/Users/ivarismartinez/Desktop/Astro_Project_2/spectrum_l=291_b=-1.2.txt', # done
                  #   '/Users/ivarismartinez/Desktop/Astro_Project_2/spectrum_l=301_b=-0.2.txt', # done
                  #   '/Users/ivarismartinez/Desktop/Astro_Project_2/spectrum_l=311_b=0.txt', # done
                  #   '/Users/ivarismartinez/Desktop/Astro_Project_2/spectrum_l=318_b=-1.5.txt', # done
                    '/Users/ivarismartinez/Desktop/Astro_Project_2/spectrum_l=328_b=1.txt', # done
                  #   '/Users/ivarismartinez/Desktop/Astro_Project_2/spectrum_l=339_b=0.txt', # done
                  #   '/Users/ivarismartinez/Desktop/Astro_Project_2/spectrum_l=350_b=0.4.txt' # done
                  ])
  
for cnt, fname in enumerate(filenames):
  outfilename=fname+'.gaussfit'
  x,y,wt,hdr=getdat(fname)
  hdr=hdr.split()
  name=hdr[0].replace('_','-')
  g.gaussfit(x,y,wt,name,outfilename)

