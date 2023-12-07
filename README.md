# Awesome-Foundation-Models-for-Weather-and-Climate
[![Awesome](https://awesome.re/badge.svg)](https://awesome.re) 
![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-green) 

A professionally curated list of **Large Foundation Models for Weather and Climate Data Understanding (e.g., time-series, spatio-temporal series, video streams, graphs, and text)** with awesome resources (paper, code, data, etc.), which aims to comprehensively and systematically summarize the recent advances to the best of our knowledge.

**<font color="red">[Paper]</font>** Our survey: [Foundation Models for Weather and Climate Data Understanding: A Comprehensive Survey](https://arxiv.org/pdf/2312.03014.pdf) has appeared on arXiv, which is the first work to comprehensively and systematically summarize DL-based weather and climate data understanding, paving the way for the development of weather and climate foundation models. 🌤️⛈️❄️


We will continue to update this list with the newest resources. If you find any missed resources (paper/code) or errors, please feel free to open an issue or make a pull request.


#### Large Foundation Models for Weather and Climate
* Accurate Medium-Range Global Weather Forecasting with 3D Neural Networks, in *Nature* 2023. [\[paper\]](https://www.nature.com/articles/s41586-023-06185-3) [\[official code\]](https://github.com/198808xc/Pangu-Weather)
* ClimaX: A Foundation Model for Weather and Climate, in *ICML* 2023. [\[paper\]](https://arxiv.org/abs/2301.10343) [\[official code\]](https://github.com/microsoft/ClimaX)
* GraphCast: Learning Skillful Medium-Range Global Weather Forecasting, in *arXiv* 2022. [\[paper\]](https://arxiv.org/abs/2212.12794) [\[official code\]](https://github.com/google-deepmind/graphcast)
* FourCastNet: A Global Data-driven High-resolution Weather Model using Adaptive Fourier Neural Operator, in *arXiv* 2022. [\[paper\]](https://arxiv.org/abs/2202.11214) [\[official code\]](https://github.com/NVlabs/FourCastNet).
* W-MAE: Pre-Trained Weather Model with Masked Autoencoder for Multi-Variable Weather Forecasting, in *arXiv* 2023. [\[paper\]](https://arxiv.org/pdf/2304.08754.pdf)  [\[official code\]](https://github.com/Gufrannn/W-MAE)
* FengWu: Pushing the Skillful Global Medium-range Weather Forecast beyond 10 Days Lead, in *arXiv* 2023. [\[paper\]](https://arxiv.org/abs/2304.02948)
* FuXi: A cascade machine learning forecasting system for 15-day global weather forecast, in *arXiv* 2023. [\[paper\]](https://arxiv.org/abs/2306.12873) [\[official code\]](https://github.com/tpys/FuXi)
* OceanGPT: A Large Language Model for Ocean Science Tasks, in *arXiv* 2023. [\[paper\]](https://arxiv.org/abs/2310.02031) [\[official code\]](https://huggingface.co/zjunlp/OceanGPT-7b)

---
  
#### Task-Specific Models for Weather and Climate
<em>Note that in this categorization, we use basic network architectures (e.g., RNN, Transformer), etc., and applications (e.g., prediction, weather pattern understanding, etc.) to make an enumeration of advanced related work.</em>

Recurrent Neural Network-based Models
* Motionrnn: A flexible model for video prediction with spacetime-varying motions, in *CVPR* 2021.
* Convolutional lstm network: A machine learning approach for precipitation nowcasting, in *NeurIPS* 2015.
* Dwfh: An improved data-driven deep weather forecasting hybrid model using transductive long short term memory (t-lstm), in *EAAI* 2023.
* Spatiotemporal inference network for precipitation nowcasting with multi-modal fusion, in *IEEE Journal of Selected Topics in Applied Earth Observation and Remote Sensing* 2023.
* Understanding the role of weather data for earth surface forecastingusing a convlstm-based model, in *CVPR* 2023.
* Spatio-temporal weather forecasting and attention mechanism on convolutional lstms, in *arXiv* 2021.
* Convolutional tensor-train lstm for spatio-temporal learning, in *NeurIPS* 2020.
* Predrnn: A recurrent neural network for spatiotemporal predictive learning, in *IEEE T-PAMI* 2022.
* Eidetic 3d lstm: A model for video prediction and beyond, in *ICLR* 2018.
* Predrann: the spatiotemporal attention convolution recurrent neural network for precipitation nowcasting, in *Knowledge-Based Systems* 2022.
* Time-series prediction of hourly atmospheric pressure using anfis and lstm approaches, in *Neural Computing and Applications* 2022.
* Ilf-lstm: Enhanced loss function in lstm to predict the sea surface temperature, in *Soft Computing* 2022.
* Swinlstm: Improving spatiotemporal prediction accuracy using swin transformer and lstm, in *ICCV* 2023.
* Swinrdm: integrate swinrnn with diffusion model towards high-resolution and highquality weather forecasting, in *AAAI* 2023.
* Swinvrnn: A data-driven ensemble forecasting model via learned distribution perturbation, in *Journal of Advances in Modeling Earth Systems* 2023.
* A generative adversarial gated recurrent unit model for precipitation nowcasting, in *IEEE Geoscience and Remote Sensing Letters* 2019.
