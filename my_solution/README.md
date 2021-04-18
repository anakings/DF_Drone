Beamforming (BF) appropriately weights the amplitude and phase of individual antenna signals to create narrowly focused radiation. This makes it possible to provide better coverage in an indoor environment and at the edge of a cell. To make the best use of this technology it is important to know the location of the device to direct the antenna beam of the radio Base Station (BS). Consequently, the Direction of Arrival (DOA) method is becoming very crucial and essential in this time.

An implementation a Machine Learning (ML) based approach for DOA by evaluating three models: Support Vector Classification (SVC), Decision Tree (DT) models and Bagging Classifier (BC). These models are trained using a public database built from drone’s radio frequency signals.

The proposed DF scheme is practical and low-complex in the sense that a phase synchronization mechanism, an antenna calibration mechanism, and the analytical model of the antenna radiation pattern are not essential. Also, the proposed DF method can be implemented using a single-channel RF receiver.

### Dependencies

- sklearn
- numpy

### Dataset

A public database is used to demonstrate our method
Dround_Data_New/Normalized

data are categorized according to 45 degree sectors
eg : 'deg_0_normalize.csv' data file represent the training data collected from the first sector and like wise there are 8 sectors considered for this study

For more details, please see the paper below.

### File description 

- get_csv_data.py : Data handler
- mysolution : Analysis of the parameters of the used models. Training size analysis. Model training and prediction


### Citation

If this is useful for your work, please cite our [Arxiv paper](https://arxiv.org/pdf/1712.01154.pdf):

```bibtex
@article{abeywickrama2017rf,
  title={RF-Based Direction Finding of UAVs Using DNN},
  author={Abeywickrama, Samith and Jayasinghe, Lahiru and Fu, Hua and Yuen, Chau},
  journal={arXiv preprint arXiv:1712.01154},
  year={2017}
}
```
