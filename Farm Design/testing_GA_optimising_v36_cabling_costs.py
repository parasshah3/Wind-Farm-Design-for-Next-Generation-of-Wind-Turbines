import numpy as np
import matplotlib.pyplot as plt
from farm_design_functions_v23 import calculate_average_turbine_spacing

layout1 = {1: (377272.2528747942, 5886263.664563148), 2: (375204.4760135094, 5886833.093819909), 3: (377871.19536309224, 5887365.467624435), 4: (377909.1291739124, 5888095.364525757), 5: (374347.0757974424, 5891605.215530233), 6: (375649.14816940605, 5891488.666126941), 7: (374366.5513417792, 5890359.949940938), 8: (378584.9199880217, 5886913.389543107), 9: (378842.35228244803, 5888715.276776684), 10: (374650.0684027991, 5889206.664322427), 11: (376682.3836407459, 5887723.698992981), 12: (378360.8001058798, 5887623.602404205), 13: (377810.04465349496, 5886444.004130621), 14: (375338.21999695833, 5887839.335255431), 15: (377001.96619372844, 5888802.192718852), 16: (377293.1470273388, 5890442.320793001), 17: (377678.7382401195, 5888877.675874941), 18: (379733.57817094185, 5886167.7740405435), 19: (377142.195906001, 5889629.5991235785), 20: (378298.0580802866, 5888643.653900515), 21: (373346.2615919043, 5892295.487248244), 22: (376454.040087674, 5888505.912590567), 23: (374994.39941858687, 5887361.153676301), 24: (376193.7079441832, 5886821.992879395), 25: (374872.612830852, 5888371.5027908785), 26: (372504.5437365566, 5891300.564753429), 27: (373731.79111258656, 5889798.694204343), 28: (376726.1488517944, 5889975.120213846), 29: (375584.534262112, 5888965.726632598), 30: (376490.28646591684, 5890990.854657781), 31: (380355.6992927018, 5884615.500801472), 32: (375844.0949513356, 5889683.79255117), 33: (373583.62628604274, 5890866.142607136), 34: (377114.3660447173, 5887453.2415373335), 35: (379197.23718007543, 5887320.385982127), 36: (376035.3120832678, 5889195.721458465), 37: (375809.8744850027, 5888118.760813552), 38: (373433.62021506124, 5888893.886848867), 39: (376207.78722803714, 5887566.702691723), 40: (374322.437252948, 5888141.435377934), 41: (376024.00537179183, 5890209.86826675), 42: (372877.5238848558, 5891682.579887656), 43: (378641.72553585726, 5886294.752958428), 44: (374356.0790362098, 5887331.624958173), 45: (373740.72572003066, 5891403.102568837), 46: (374039.8606145159, 5892127.920194798), 47: (377834.1005418886, 5890248.776591203), 48: (377247.1449823146, 5886778.4234297285), 49: (372724.38883389754, 5890556.287512451), 50: (375324.2054758389, 5890528.648001432), 51: (378571.1188297298, 5889872.774811584), 52: (372467.34654026607, 5892272.053773943), 53: (376607.51304446935, 5886309.361673354), 54: (374380.0360693898, 5889644.027138722), 55: (378948.54320141335, 5885215.4050696865), 56: (374025.0239606977, 5888874.415722471), 57: (377015.77952437854, 5890969.823447297), 58: (373725.8429649261, 5887763.704353819), 59: (375220.4889893406, 5891126.639715698), 60: (372890.31151972717, 5889729.084391208), 61: (375194.6879371266, 5889307.068676184), 62: (371755.4969391172, 5893134.750461282), 63: (378358.6450253866, 5890485.430714468), 64: (379428.0679003396, 5885376.931380787), 65: (378233.111542457, 5889470.510135399), 66: (375583.69606164575, 5887256.192478023), 67: (378598.4971829129, 5885747.0925241085), 68: (376654.96831417433, 5889378.198745781), 69: (373454.0439975902, 5888191.847605424), 70: (379119.018724485, 5888205.323592609), 71: (375204.58988586266, 5889928.2721929345), 72: (371910.74086384836, 5892627.771590184), 73: (379463.715182091, 5886671.275644124), 74: (374479.44283770956, 5891120.22915155), 75: (380139.99150309304, 5885274.875936779), 76: (373357.7481881132, 5889400.999667835), 77: (376736.9151097158, 5886978.024978797), 78: (372771.02076151397, 5892763.0883527985), 79: (375054.80232321133, 5891627.4374510385), 80: (373293.91480553854, 5890235.937770181), 81: (377628.8870771721, 5885772.912213577), 82: (374644.7245583233, 5892087.232810551), 83: (377435.45631942863, 5888399.986954547), 84: (372314.5779851341, 5893080.350434035), 85: (377707.56652417366, 5889687.4191182805), 86: (372181.50473205117, 5891734.193000662), 87: (376884.0367734783, 5888232.559376976), 88: (375717.43228537013, 5890888.545278477), 89: (373002.9091166416, 5891155.409851936), 90: (375412.6930806188, 5888491.184584449), 91: (378084.06482052297, 5885987.1392145865), 92: (376571.9971586469, 5890457.892107408), 93: (378821.0489650679, 5889317.859546087), 94: (379754.5773877143, 5884895.391532061), 95: (379107.40716517624, 5885793.438556471), 96: (378892.3203292586, 5887737.425013122), 97: (374750.1578531177, 5887876.851700411)}
layout2 = {1: (375218.5670106594, 5889749.825712176), 2: (372583.67168008164, 5891725.01670601), 3: (376775.2892357752, 5888643.455805937), 4: (379934.99252559664, 5885236.531479083), 5: (378760.46459137986, 5888390.6140385065)}
layout3 = {20: (376748.6206005998, 5888173.007509838), 21: (374578.0078346767, 5889536.593157514), 22: (376281.28916444513, 5889190.984730546), 32: (376040.7706342464, 5887975.15167003), 33: (375303.83130127646, 5889592.638130531), 35: (375218.44878831477, 5887824.6527444), 94: (376902.82513581734, 5889207.6523396475), 95: (377264.2602922631, 5887998.334280464), 109: (376991.9295583247, 5889783.056154078), 129: (376923.6998911574, 5886943.851879547), 139: (375381.92911492207, 5887036.493298339), 140: (375972.73186564684, 5887125.9865509), 141: (376522.2783395304, 5886409.604873156), 143: (375382.6280525106, 5890443.196900563), 150: (373748.0401951036, 5887970.049032408), 158: (376673.51068616705, 5891196.613647437), 159: (375132.30966868106, 5891729.517387884), 160: (373584.138227372, 5892119.836924347), 161: (378255.17474531836, 5889184.92811884), 162: (377560.571560382, 5888995.9302916825), 163: (378119.09242473746, 5890151.415752356), 164: (373390.71851905755, 5889330.633515005), 165: (378804.117158024, 5887924.624404422), 166: (373424.4472669403, 5891418.636346222), 167: (378132.17752403044, 5887161.601959413), 168: (377666.25136742013, 5886721.5170308165), 169: (373065.86803177965, 5889709.515725955), 170: (379627.0658682275, 5887187.673736717), 171: (378311.6109091856, 5886027.893088668), 172: (379153.6999091905, 5885210.592649156), 173: (379344.05878020066, 5886120.618840538), 174: (372788.62984271266, 5892673.9326348035), 175: (380085.2793206433, 5884808.268758974), 176: (379819.49016086303, 5885769.054087366), 177: (372584.7724586973, 5890725.139691089), 178: (371973.9364479561, 5892709.618724223), 179: (372196.28427010484, 5892112.178835784), 180: (374155.339589658, 5889075.481080425), 181: (378860.53568905056, 5889255.588598089)}

average_spacing1,_,_,_ = calculate_average_turbine_spacing(layout1)
average_spacing_rotor_diameters1 = average_spacing1 / (2 * 124.7)
print(f"Average turbine spacing 1: {average_spacing1:.2f} m")
print(f"Average turbine spacing in rotor diameters 1: {average_spacing_rotor_diameters1:.2f}")

average_spacting2,_,_,_ = calculate_average_turbine_spacing(layout2)
average_spacing_rotor_diameters2 = average_spacting2 / (2 * 124.7)
print(f"Average turbine spacing 2: {average_spacting2:.2f} m")
print(f"Average turbine spacing in rotor diameters 2: {average_spacing_rotor_diameters2:.2f}")

average_spacing3,_,_,_ = calculate_average_turbine_spacing(layout3)
average_spacing_rotor_diameters3 = average_spacing3 / (2 * 124.7)
print(f"Average turbine spacing 3: {average_spacing3:.2f} m")
print(f"Average turbine spacing in rotor diameters 3: {average_spacing_rotor_diameters3:.2f}")
