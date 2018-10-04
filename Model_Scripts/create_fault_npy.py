import numpy as np
import xlrd

outfile = 'fault_array.npy'

workbook = xlrd.open_workbook('Fixed92kmFaultOffset50kmgapPts.xls')

worksheet = workbook.sheet_by_name('Fixed92kmFaultOffset50kmgapPts')

fault_array = np.zeros(0)

for i in range(1,29):
	x = worksheet.cell(i, 2).value
	y = worksheet.cell(i, 3).value
	temp_array = np.array([x,y])
	fault_array = np.append(fault_array, temp_array)

np.save(outfile, fault_array)