from final_model.src.fusion import AddressFusionEngine

eng = AddressFusionEngine()
df = eng.spatial_poi_df
t1 = df[(df['town_id'] == 'T1') & (df['landmark_type'] == 'post_office')]
print(t1[['poi_id', 'name', 'locality_id', 'x', 'y']].to_string())
print("Locality ids in T1 post office rows:", t1['locality_id'].unique())
print("Looking for: T1-L03")