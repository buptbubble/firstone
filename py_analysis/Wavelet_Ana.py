import pywt

class wavelet_ana:

    wavelet_type = 'db3'
    wavelet_level = 4

    def get_wavelet_coeffs(self,val_list):
        coeffs = pywt.wavedec(val_list,mode = self.wavelet_type,level = self.wavelet_level)
        return coeffs

    def reconstruction_from_coeffs(self,coeffs):
        return pywt.waverec(coeffs,mode= self.wavelet_type)

    def
