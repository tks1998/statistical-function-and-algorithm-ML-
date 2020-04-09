import numpy as np
import math
from statistics import median
from scipy.stats import skew
import weightedstats as ws
from statsmodels.stats.stattools import medcouple

class Med_couple:
    
    def __init__(self,data):
        self.data = np.sort(data,axis = None)[::-1] # sorted decreasing  
        self.med = np.median(self.data)
        self.scale = 2*np.amax(np.absolute(self.data))
        self.Zplus = [(x-self.med)/self.scale for x in self.data if x>=self.med]
        self.Zminus = [(x-self.med)/self.scale for x in self.data if x<=self.med]
        self.p = len(self.Zplus)
        self.q = len(self.Zminus)
    
    def H(self,i,j):
        a = self.Zplus[i]
        b = self.Zminus[j]

        if a==b:
            return np.sign(self.p - 1 - i - j)
        else:
            return (a+b)/(a-b)

    def greater_h(self,u):

        P = [0]*self.p

        j = 0

        for i in range(self.p-1,-1,-1):
            while j < self.q and self.H(i,j)>u:
                j+=1
            P[i]=j-1
        return P

    def less_h(self,u):

        Q = [0]*self.p

        j = self.q - 1

        for i in range(self.p):
            while j>=0 and self.H(i,j) < u:
                j=j-1
            Q[i]=j+1
        
        return Q
    #Kth pair algorithm (Johnson & Mizoguchi)
    def kth_pair_algorithm(self):
        L = [0]*self.p
        R = [self.q-1]*self.p

        Ltotal = 0

        Rtotal = self.p*self.q

        medcouple_index = math.floor(Rtotal / 2)

        while Rtotal - Ltotal > self.p:

            middle_idx = [i for i in range(self.p) if L[i]<=R[i]]
            row_medians = [self.H(i,math.floor((L[i]+R[i])/2)) for i in middle_idx]

            weight = [R[i]-L[i] + 1 for i in middle_idx]

            WM = ws.weighted_median(row_medians,weights = weight)
            
            P = self.greater_h(WM)

            Q = self.less_h(WM)

            Ptotal = np.sum(P)+len(P) 
            Qtotal = np.sum(Q)

            if medcouple_index <= Ptotal-1:
                R = P.copy()
                Rtotal = Ptotal
            else:
                if medcouple_index > Qtotal - 1:
                    L = Q.copy()
                    Ltotal = Qtotal
                else:
                    return WM
        remaining = np.array([])
       
        for i in range(self.p):
            for j in range(L[i],R[i]+1):
                remaining = np.append(remaining,self.H(i,j))

        find_index = medcouple_index-Ltotal

        k_minimum_element = remaining[np.argpartition(remaining,find_index)]
        
        # print(find_index,'tim trong mang ',sorted(remaining))
        return k_minimum_element[find_index]
       
    def naive_algorithm_testing(self):
        result = [self.H(i,j) for i in range(self.p) for j in range(self.q)]
        return np.median(result)

if __name__ == '__main__':
    sum=0
    for i in range(1000):
        data = np.random.randint(low = 0, high = 200000, size = 1000) 

        A = Med_couple(data)
        sum+=abs(medcouple(data)-A.kth_pair_algorithm())
        # print(skew(data))
        # print("kth",A.kth_pair_algorithm())
        # print("naive my code",A.naive_algorithm_testing())
        # print("naive",medcouple(data))
    print(sum)