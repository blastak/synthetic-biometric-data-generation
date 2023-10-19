import ctypes as c

# mydll = c.WinDLL(r'C:\Neurotec_Biometric_12_4_SDK\Bin\Win64_x64\Neurotec.Licensing.dll')
# myfunc = mydll['NLicense']
# # myfunc = mydll['Neurotec.Licensing.NLicense']
# # myfunc = mydll['Neurotec.Licensing.NLicense']
# var = myfunc.ObtainComponents("/local", 5000, "Biometrics.IrisExtraction,Biometrics.IrisMatching")
# print(var)
#


import os
import clr

license_path = r'C:\Neurotec_Biometric_12_4_SDK\Bin\Win64_x64'

clr.AddReference(os.path.join(license_path, 'Neurotec.dll'))
clr.AddReference(os.path.join(license_path, 'Neurotec.Biometrics.dll'))
clr.AddReference(os.path.join(license_path, 'Neurotec.Biometrics.Client.dll'))
clr.AddReference(os.path.join(license_path, 'Neurotec.Licensing.dll'))
clr.AddReference(os.path.join(license_path, 'Neurotec.Media.dll'))
# from Neurotec.Licensing import NLicense #### success
# import Neurotec.Licensing.NLicense as NLicense #### error
Neurotec = __import__('Neurotec')
# NLicense = Neurotec.Licensing.NLicense
var = Neurotec.Licensing.NLicense.ObtainComponents("/local", 5000, "Biometrics.IrisExtraction,Biometrics.IrisMatching")
print(var)