import xlwings as xw
import numpy as np
from farm_design_functions_v28 import LCOE

def evaluate_layout(layout, file_path, sheet_name, Ct, r_0, z, z_0, wind_rose):
    """
    Evaluates the LCOE and energy yield of a given wind farm layout.

    Parameters:
        layout (dict): Dictionary of turbine positions {id: (x, y)}.
        file_path (str): Path to the Excel model.
        sheet_name (str): Sheet name in the Excel file.
        Ct, r_0, z, z_0: Turbine and wind profile parameters.
        wind_rose (dict): Wind direction-frequency data.

    Returns:
        tuple: (LCOE in £/MWh, annual energy yield in MWh)
    """
    # Start Excel app
    app = xw.App(visible=False, add_book=False)
    app.display_alerts = False
    app.screen_updating = False

    wb = xw.Book(file_path)
    sheet = wb.sheets[sheet_name]

    # Evaluate using existing function
    lcoe = LCOE(file_path, sheet_name, Ct, r_0, z, z_0, layout, wind_rose)

    # Clean up
    wb.close()
    app.quit()

    return lcoe


# ---- Example Usage ----
if __name__ == "__main__":
    file_path = "/Users/paras/Desktop/3YP Python Scripts/Farm Design/Wind Farms .csv Files/25MW_EighthIteration.xlsm"
    sheet_name = "WindDist"

    # Example layout: {id: (x, y)}
    layout = {1: (379666.8050880782, 5885177.402653365), 2: (372740.554264809, 5892199.31177809), 3: (379142.3145709715, 5887119.628671083), 4: (374069.27534953464, 5887885.997324014), 5: (373854.8983216913, 5889310.876991706), 6: (376378.43200445693, 5891244.456032305), 7: (378090.30378801905, 5890559.359605563), 8: (378814.9371250947, 5888222.461630864), 9: (373962.93023924984, 5891302.060447295), 10: (378450.70102554606, 5887445.319626779), 11: (376987.61542937724, 5888066.203701933), 12: (377298.0897873416, 5890498.909701263), 13: (372919.9683406212, 5890328.797407054), 14: (376614.9427260884, 5887059.619239042), 15: (378420.35733427946, 5886391.033194531), 16: (375257.46832144, 5890294.135035423), 17: (377480.5624448201, 5888916.58759064), 18: (374946.981243862, 5891619.133588196), 19: (374708.737357506, 5888381.106823244), 20: (375082.5441898906, 5889125.37585684), 21: (375531.01901271654, 5888417.308764907), 22: (371866.03934140626, 5892890.25456724), 23: (377709.1574954812, 5887094.522035804), 24: (375316.58433553035, 5886792.569693681), 25: (378578.6768018719, 5889753.374633078), 26: (374037.26475080685, 5892138.04447283), 27: (375610.05368325545, 5887537.628763694), 28: (375913.7237512283, 5889495.48723136), 29: (378068.24167911697, 5888373.618568934), 30: (376963.9235221979, 5886204.633297758), 31: (379230.61274744105, 5886294.663733808), 32: (378870.103391475, 5885400.811437494), 33: (372519.82428446907, 5891273.294179768), 34: (376587.1652227918, 5890080.506450335), 35: (374856.49154837074, 5887564.715893765), 36: (373039.85335788026, 5889274.4073978495), 37: (374323.97991704853, 5890117.081913754), 38: (379877.1257609006, 5885915.450703345), 39: (376450.4051032231, 5888764.741381107), 40: (377593.79922847665, 5889809.867656554)}
    Ct = 0.790
    r_0 = 124.7
    z = 140
    z_0 = 7.8662884488e-04
    wind_rose = {
        0: 0.06, 30: 0.03, 60: 0.04, 90: 0.06, 120: 0.05, 150: 0.07,
        180: 0.09, 210: 0.17, 240: 0.19, 270: 0.12, 300: 0.07, 330: 0.05
    }

    lcoe, energy = evaluate_layout(layout, file_path, sheet_name, Ct, r_0, z, z_0, wind_rose)
    print(f"LCOE: {lcoe:.2f} £/MWh")
    print(f"Annual Energy Yield: {energy:.2f} MWh")