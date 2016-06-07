import pywt
import heapq
import numpy as np

class wavelet_ana:

    wavelet_type = 'db3'
    wavelet_level = 4

    def get_wavelet_coeffs(self,val_list):
        coeffs = pywt.wavedec(val_list,self.wavelet_type,level = self.wavelet_level)
        return coeffs

    def reconstruction_from_coeffs(self,coeffs):
        return pywt.waverec(coeffs,self.wavelet_type)

    def coeffs_process(self,coeffs):
       # coeffs_out = coeffs
        for level in range(len(coeffs)):
            if level == 0:
                continue
            if level == 1:
                std = np.std(coeffs[level])
                #coeffs[level]=self.retain_nmax_coeffs(3,coeffs[level])
                coeffs[level]=pywt.threshold(coeffs[level],std,'hard')

                continue
            if level == 2:
                #coeffs[level] = self.retain_nmax_coeffs(3, coeffs[level])
                std = np.std(coeffs[level])
                coeffs[level] = pywt.threshold(coeffs[level], std*1.5, 'hard')
                continue
            if level == 3:
                # coeffs[level] = self.retain_nmax_coeffs(3, coeffs[level])
                std = np.std(coeffs[level])
                coeffs[level] = pywt.threshold(coeffs[level], std * 2, 'hard')
                continue
            else:
                coeffs[level] = self.retain_nmax_coeffs(0, coeffs[level])
        return coeffs


    def retain_nmax_coeffs(self,n,coeffs):
        c_temp = [abs(x) for x in coeffs]
        c_largest = heapq.nlargest(n,c_temp)

        c_result = np.zeros(len(c_temp))
        for c in c_largest:
            idx = c_temp.index(c)
            c_result[idx] = coeffs[idx]
        return  c_result

if __name__ == '__main__':
    wa = wavelet_ana()

    a = [1,2,3,4,5,-6,7]
    print(wa.retain_nmax_coeffs(0,a))



