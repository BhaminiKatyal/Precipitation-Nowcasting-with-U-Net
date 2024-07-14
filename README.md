# Precipitation Nowcasting with U-Net

Heavy rainfall events in urban areas pose a growing threat to public safety and infrastructure. Issuing timely warnings for these events is crucial, but short-term weather forecasting (nowcasting) can be challenging. This project investigates U-Net's potential for precipitation nowcasting, providing valuable information for issuing critical warnings and enabling proactive measures.

### Dataset used: 
GPM IMERG Final Precipitation L3 Half Hourly 0.1 degree x 0.1 degree V07 <br>
(https://disc.gsfc.nasa.gov/datasets/GPM_3IMERGHH_07/summary?keywords=imerg) <br>

Data Region (Latitudes and Longitudes): 69.422 W, 19.881 S, 85.022 E, 35.422 N <br>
Data Range: 1st July 2023 to 15th September 2023 <br>
Variables: Precipitation <br>
File Format: netCDF

### Model Input and Preprocessing
This section details the preparation of GPM IMERG data for training the U-net model in time-series precipitation prediction. The approach leverages the temporal relationships between precipitation events. <br><br>
<b> Sequence-based Prediction:</b><br>
The model utilizes a sequence-based approach, exploiting the temporal dimension of the data. This involves using the previous 3 timestamps' precipitation data (each representing 30 minutes) to predict the precipitation for the next timestamp (another 30 minutes).
To achieve this, 3 half-hourly precipitation images are concatenated into a single tensor with a dimension of 3x128x128. Each channel in this tensor represents the precipitation data from a different previous time step. This provides the model with a short-term history of precipitation events (covering the past 1.5 hours).

### Model:
This project utilizes a U-net architecture with an encoder-decoder structure. ReLU activations with batch normalization are used throughout for non-linearity and training stability. He initialization ensures proper training for ReLU layers.<br>
<br> The architecture used for the model is similar to the below image. <br>
![image](https://github.com/user-attachments/assets/9bc33a4d-9b52-4e67-bb79-365e0ce6194d)

### Training:
The model was trained on a dataset of 619 time-series precipitation samples. Here are the key training parameters: <br>
•	Batch Size: 8 images <br>
•	Epochs: 15 <br>
•	Loss Function: Mean Squared Error (MSE) <br>
•	Optimizer: Stochastic Gradient Descent (SGD) 

![image](https://github.com/user-attachments/assets/a9a7d943-7026-4c01-ba87-13103bafbf59)


### Results:
The results were obtained as follows: <br>
![image](https://github.com/user-attachments/assets/86397485-13c3-47c6-88ae-1085a65f7c73) <br>
<i>i) 3D input image  ii) actual mask iii) model’s output </i><br>
<br>
Root Mean Squared Error (RMSE): 0.8 mm/hr <br>
Pearson Correlation Coefficient (PCC): 0.78 <br>

### Acknowledgements:
I would like to thank Dr. Shruti Upadhyaya (Assistant Professor, Dept of Civil Engineering and Climate Change, IIT Hyderabad) for her guidance and support throughout this internship project. Her expertise and insights were invaluable in helping me achieve these results. I would also like to express my gratitude to Mr. Abhigyan Chakraborty and Mr. Yashraj Nagraj Upase of the Climate Change Department at IITH. Their assistance with data acquisition was instrumental to this project.






