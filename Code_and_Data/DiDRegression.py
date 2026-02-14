import os
import pandas as pd
import numpy as np
import statsmodels.api as sm

# Get current folder (same as where this script is saved)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Explicitly list the Excel files
excel_files = [
    "India_Exports_Mangoes_to_Mauritius_and_SouthAfrica.xlsx",
    "India_Imports_Pigeon_Peas_from_Malawi_and_Mozambique.xlsx"
]

for file in excel_files:
    file_path = os.path.join(current_dir, file)
    
    print("="*100)
    print(f"📘 Running regressions for file: {file}")
    print("="*100)
    
    # Read 'Regression' sheet
    df = pd.read_excel(file_path, sheet_name='Regression')
    
    # Drop rows with missing values in relevant columns (ensure both regressions can run)
    df = df.dropna(subset=[
        'Trade Value',
        'Producer Prices', 
        'Country Dummy', 
        'Post MoU Dummy', 
        'MoU in Effect Dummy', 
        'Export Share', 
        'GPR_world',
        'GPR_importer',
        'GPR_exporter',
        'Import Share',
        'PS_exporter',
        'PS_importer',
        'MoU * Import Share'
    ]).copy()
    
    # Guard: ensure Trade Value > 0 before log
    if (df['Trade Value'] <= 0).any():
        eps = 1e-6
        df['Trade Value'] = df['Trade Value'].clip(lower=eps)
    
    # Take natural log of Trade Value
    df['Log_Trade_Value'] = np.log(df['Trade Value'])
    
    # -----------------------
    # Regression 1: DV = Log_Trade_Value
    # IVs: Country Dummy, Post MoU Dummy, MoU in Effect Dummy, Import Share, PS_importer
    # -----------------------
    print("\n🍋 Regression 1: DV = Log_Trade_Value")
    y1 = df['Log_Trade_Value']
    X1 = df[['Country Dummy', 'Post MoU Dummy', 'MoU in Effect Dummy', 'Import Share', 'GPR_importer']]
    X1 = sm.add_constant(X1, has_constant='add')
    try:
        model1 = sm.OLS(y1, X1).fit()
        print(model1.summary())
    except Exception as e:
        print(f"⚠️ Error fitting Regression 1 on {file}: {e}")

    # -----------------------
    # Regression 2: DV = Producer Prices
    # IVs: Country Dummy, Post MoU Dummy, MoU in Effect Dummy, Import Share, GPR_importer
    # -----------------------
    print("\n🔶 Regression 2: DV = Producer Prices")
    y2 = df['Producer Prices']
    X2 = df[['Country Dummy', 'Post MoU Dummy', 'MoU in Effect Dummy', 'Import Share', 'GPR_importer']]
    X2 = sm.add_constant(X2, has_constant='add')
    try:
        model2 = sm.OLS(y2, X2).fit()
        print(model2.summary())
    except Exception as e:
        print(f"⚠️ Error fitting Regression 2 on {file}: {e}")

    print("\n\n")

print("✅ All regressions completed.")
