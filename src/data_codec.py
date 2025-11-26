from sklearn.preprocessing import LabelEncoder

def encode(raw_df, encoded_path="data/encoded_data.csv"):
    #set variables 
    encoders = {}
    encoded_df = raw_df.copy()
    
    #encode values 
    for column in encoded_df:
        encoder = LabelEncoder()
        encoded_df[column] = encoder.fit_transform(encoded_df[column].astype(str))
        encoders[column] = encoder
        
    # Save encoded data
    encoded_df.to_csv(encoded_path, index=False)
    
    return encoded_df, encoders