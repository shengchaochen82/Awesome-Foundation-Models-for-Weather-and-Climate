# Awesome-Foundation-Models-for-Weather-and-Climate
[![Awesome](https://awesome.re/badge.svg)](https://awesome.re) 
![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-green) 
![Stars](https://img.shields.io/github/stars/shengchaochen82/Awesome-Foundation-Models-for-Weather-and-Climate)

A professionally curated list of **Large Foundation Models for Weather and Climate Data Understanding (e.g., time-series, spatio-temporal series, video streams, graphs, and text)** with awesome resources (paper, code, data, etc.), which aims to comprehensively and systematically summarize the recent advances to the best of our knowledge.

**<font color="red">[Paper]</font>** Our survey: [Foundation Models for Weather and Climate Data Understanding: A Comprehensive Survey](https://arxiv.org/pdf/2312.03014.pdf) has appeared on arXiv, which is the first work to comprehensively and systematically summarize DL-based weather and climate data understanding, paving the way for the development of weather and climate foundation models. 🌤️⛈️❄️


We will continue to update this list with the newest resources. If you find any missed resources (paper/code) or errors, please feel free to open an issue or make a pull request.

___
## Large Foundation Models for Weather and Climate
>**Definition**:
*Pre-trained from large-scale weather/climate dataset and able to perform various weather/cliamte-related tasks.*

* Pangu-Weather: Accurate Medium-Range Global Weather Forecasting with 3D Neural Networks, in *Nature* 2023. [\[paper\]](https://www.nature.com/articles/s41586-023-06185-3) [\[official code\]](https://github.com/198808xc/Pangu-Weather)
* ClimaX: A Foundation Model for Weather and Climate, in *ICML* 2023. [\[paper\]](https://arxiv.org/abs/2301.10343) [\[official code\]](https://github.com/microsoft/ClimaX)
* GraphCast: Learning Skillful Medium-Range Global Weather Forecasting, in *arXiv* 2022. [\[paper\]](https://arxiv.org/abs/2212.12794) [\[official code\]](https://github.com/google-deepmind/graphcast)
* FourCastNet: A Global Data-driven High-resolution Weather Model using Adaptive Fourier Neural Operator, in *arXiv* 2022. [\[paper\]](https://arxiv.org/abs/2202.11214) [\[official code\]](https://github.com/NVlabs/FourCastNet)
* W-MAE: Pre-Trained Weather Model with Masked Autoencoder for Multi-Variable Weather Forecasting, in *arXiv* 2023. [\[paper\]](https://arxiv.org/abs/2304.08754)  [\[official code\]](https://github.com/Gufrannn/W-MAE)
* FengWu: Pushing the Skillful Global Medium-range Weather Forecast beyond 10 Days Lead, in *arXiv* 2023. [\[paper\]](https://arxiv.org/abs/2304.02948)
* FuXi: A cascade machine learning forecasting system for 15-day global weather forecast, in *arXiv* 2023. [\[paper\]](https://arxiv.org/abs/2306.12873) [\[official code\]](https://github.com/tpys/FuXi)
* OceanGPT: A Large Language Model for Ocean Science Tasks, in *arXiv* 2023. [\[paper\]](https://arxiv.org/abs/2310.02031) [\[official code\]](https://huggingface.co/zjunlp/OceanGPT-7b)

---
  
## Task-Specific Models for Weather and Climate
> **Remark**: *Note that in this categorization, we use basic network architectures (e.g., RNN, Transformer), etc., and applications (e.g., prediction, weather pattern understanding, etc.) to make an enumeration of advanced related work.*

**Recurrent Neural Network-based Models**

* MotionRNN: A Flexible Model for Video Prediction with Spacetime-Varying Motions, in *CVPR* 2021. [\[Paper\]](https://arxiv.org/abs/2103.02243) [\[official code\]](https://github.com/thuml/MotionRNN)
* Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting, in *NeurIPS* 2015. [\[Paper\]](https://proceedings.neurips.cc/paper/2015/hash/07563a3fe3bbe7e3ba84431ad9d055af-Abstract.html) [\[official code\]](https://github.com/ndrplz/ConvLSTM_pytorch) 
* Dwfh: An improved data-driven deep weather forecasting hybrid model using transductive long short term memory (t-lstm), in *EAAI* 2023. [\[Paper\]](https://www.sciencedirect.com/science/article/pii/S0957417422022886)
* Spatiotemporal inference network for precipitation nowcasting with multi-modal fusion, in *IEEE Journal of Selected Topics in Applied Earth Observation and Remote Sensing* 2023.[\[Paper\]](https://ieeexplore.ieee.org/abstract/document/10285341)
* Understanding the role of weather data for earth surface forecastingusing a convlstm-based model, in *CVPR* 2023. [\[Paper\]](https://openaccess.thecvf.com/content/CVPR2022W/EarthVision/papers/Diaconu_Understanding_the_Role_of_Weather_Data_for_Earth_Surface_Forecasting_CVPRW_2022_paper.pdf) [\[official code\]](https://github.com/dcodrut/weather2land)
* Spatio-temporal weather forecasting and attention mechanism on convolutional lstms, in *arXiv* 2021. [\[Paper\]](https://www.academia.edu/download/93935645/2102.00696v1.pdf) [\[official code\]](https://github.com/sftekin/spatio-temporal-weather-forecasting)
* Convolutional tensor-train lstm for spatio-temporal learning, in *NeurIPS* 2020. [\[Paper\]](https://proceedings.neurips.cc/paper/2020/hash/9e1a36515d6704d7eb7a30d783400e5d-Abstract.html) [\[official code\]](https://github.com/NVlabs/conv-tt-lstm)
* Predrnn: A recurrent neural network for spatiotemporal predictive learning, in *IEEE T-PAMI* 2022. [\[Paper\]](https://ieeexplore.ieee.org/abstract/document/9749915) [\[official code\]](https://github.com/thuml/predrnn-pytorch) 
* Eidetic 3d lstm: A model for video prediction and beyond, in *ICLR* 2018. [\[Paper\]](http://faculty.ucmerced.edu/mhyang/papers/iclr2019_eidetic3d.pdf) [\[official code\]](https://github.com/google/e3d_lstm) 
* Predrann: the spatiotemporal attention convolution recurrent neural network for precipitation nowcasting, in *Knowledge-Based Systems* 2022. [\[Paper\]](https://www.sciencedirect.com/science/article/pii/S0950705121010601) 
* Time-series prediction of hourly atmospheric pressure using anfis and lstm approaches, in *Neural Computing and Applications* 2022. [\[Paper\]](https://link.springer.com/article/10.1007/s00521-022-07275-5) 
* Ilf-lstm: Enhanced loss function in lstm to predict the sea surface temperature, in *Soft Computing* 2022. [\[Paper\]](https://link.springer.com/article/10.1007/s00500-022-06899-y) 
* Swinlstm: Improving spatiotemporal prediction accuracy using swin transformer and lstm, in *ICCV* 2023. [\[Paper\]](https://openaccess.thecvf.com/content/ICCV2023/html/Tang_SwinLSTM_Improving_Spatiotemporal_Prediction_Accuracy_using_Swin_Transformer_and_LSTM_ICCV_2023_paper.html) [\[official code\]](https://github.com/SongTang-x/SwinLSTM) 
* Swinrdm: integrate swinrnn with diffusion model towards high-resolution and highquality weather forecasting, in *AAAI* 2023. [\[Paper\]](https://ojs.aaai.org/index.php/AAAI/article/view/25105) 
* Swinvrnn: A data-driven ensemble forecasting model via learned distribution perturbation, in *Journal of Advances in Modeling Earth Systems* 2023. [\[Paper\]](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2022MS003211)
* Comparison of BLSTM-Attention and BLSTM-Transformer Models for Wind Speed Prediction, in *Bulgarian Academyof Sciences* 2022. [\[Paper\]](http://proceedings.bas.bg/index.php/cr/article/view/10)
* A generative adversarial gated recurrent unit model for precipitation nowcasting, in *IEEE Geoscience and Remote Sensing Letters* 2019. [\[Paper\]](https://ieeexplore.ieee.org/abstract/document/8777193) [\[official code\]](https://github.com/LukaDoncic0/GAN-argcPredNet) 
* Stochastic Super-Resolution for Downscaling Time-Evolving Atmospheric Fields With a Generative Adversarial Network, in *IEEE Transactions on Geoscience and Remote Sensing* 2020. [\[Paper\]](https://ieeexplore.ieee.org/abstract/document/9246532) [\[official code\]](https://github.com/jleinonen/downscaling-rnn-gan) 
* Swin Transformer: Hierarchical Vision Transformer Using Shifted Windows, in *ICCV* 2021. [\[Paper\]](https://openaccess.thecvf.com/content/ICCV2021/html/Liu_Swin_Transformer_Hierarchical_Vision_Transformer_Using_Shifted_Windows_ICCV_2021_paper) [\[official code\]](https://github.com/microsoft/Swin-Transformer) 
* Towards data-driven physics-informed global precipitation forecasting from satellite imagery, in *NeurIPS* 2020. [\[Paper\]](https://s3.us-east-1.amazonaws.com/climate-change-ai/papers/neurips2020/70/paper.pdf)

**Diffusion Models-based Approaches**
* SwinRDM: Integrate SwinRNN with Diffusion Model towards High-Resolution and High-Quality Weather Forecasting, in *AAAI* 2023. [\[Paper\]](https://ojs.aaai.org/index.php/AAAI/article/view/25105)
* Swinvrnn: A data-driven ensemble forecasting model via learned distribution perturbation, in *Journal of Advances in Modeling Earth Systems* 2023. [\[Paper\]](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2022MS003211)
* SEEDS: Emulation of Weather Forecast Ensembles with Diffusion Models, in *arXiv* 2023. [\[Paper\]](https://arxiv.org/abs/2306.14066)
* DiTTO: Diffusion-inspired Temporal Transformer Operator, in *arXiv* 2023. [\[Paper\]](https://arxiv.org/abs/2307.09072)
* PreDiff: Precipitation Nowcasting with Latent Diffusion Models, in *arXiv* 2023. [\[Paper\]](https://arxiv.org/abs/2307.10422)
* Latent diffusion models for generative precipitation nowcasting with accurate uncertainty quantification, in *arXiv* 2023. [\[Paper\]](https://arxiv.org/abs/2304.12891) [\[official code\]](https://github.com/MeteoSwiss/ldcast)
* ClimART: A Benchmark Dataset for Emulating Atmospheric Radiative Transfer in Weather and Climate Models, in *arXiv* 2021. [\[Paper\]](https://arxiv.org/abs/2111.14671) [\[official code\]](https://github.com/RolnickLab/climart)
* PDE-Refiner: Achieving Accurate Long Rollouts with Neural PDE Solvers, in *arXiv* 2023. [\[Paper\]](https://arxiv.org/abs/2308.05732)
* Diffusion Models for High-Resolution Solar Forecasts, in *arXiv* 2023. [\[Paper\]](https://arxiv.org/abs/2302.00170)
* Generative Residual Diffusion Modeling for Km-scale Atmospheric Downscaling, in *arXiv* 2023. [\[Paper\]](https://arxiv.org/abs/2309.15214)
* DiffMet: Diffusion models and deep learning for precipitation nowcasting, in *Master thesis* 2023. [\[Paper\]](https://www.duo.uio.no/handle/10852/103253)

**Generative Adversarial Networks (GANs)-based Approaches**
* Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks, in *arXiv* 2015. [\[Paper\]](https://arxiv.org/abs/1511.06434) [\[official code\]](https://github.com/Newmu/dcgan_code)
* Large Scale GAN Training for High Fidelity Natural Image Synthesis, in *arXiv* 2018. [\[Paper\]](https://arxiv.org/abs/1809.11096) [\[official code\]](https://github.com/ajbrock/BigGAN-PyTorch)
* Progressive Growing of GANs for Improved Quality, Stability, and Variation, in *arXiv* 2018. [\[Paper\]](https://arxiv.org/abs/1710.10196) [\[official code\]](https://github.com/tkarras/progressive_growing_of_gans)
* A generative adversarial network approach to (ensemble) weather prediction, in *Neural Networks* 2021. [\[Paper\]](https://www.sciencedirect.com/science/article/abs/pii/S0893608021000459)
* Climate-StyleGAN : Modeling Turbulent Climate Dynamics Using Style-GAN, in *AI for Earth Science Workshop* 2020. [\[Paper\]](https://ai4earthscience.github.io/neurips-2020-workshop/papers/ai4earth_neurips_2020_53.pdf)
* Dynamic Multiscale Fusion Generative Adversarial Network for Radar Image Extrapolation, in *IEEE Transactions on Geoscience and Remote Sensing* 2022. [\[Paper\]](https://ieeexplore.ieee.org/abstract/document/9837952)
* Generative modeling of spatio-temporal weather patterns with extreme event conditioning, in *arXiv* 2021. [\[Paper\]](https://arxiv.org/abs/2104.12469)
* Skilful precipitation nowcasting using deep generative models of radar, in *Nature* 2021. [\[Paper\]](https://www.nature.com/articles/s41586-021-03854-z) [\[official code\]](https://github.com/openclimatefix/skillful_nowcasting)
* SPATE-GAN: Improved Generative Modeling of Dynamic Spatio-Temporal Patterns with an Autoregressive Embedding Loss, in *AAAI* 2022. [\[Paper\]](https://ojs.aaai.org/index.php/AAAI/article/view/20375) [\[official code\]](https://github.com/konstantinklemmer/spate-gan)
* MPL-GAN: Toward Realistic Meteorological Predictive Learning Using Conditional GAN, in *IEEE Access* 2020. [\[Paper\]](https://ieeexplore.ieee.org/abstract/document/9094665)
* PCT-CycleGAN: Paired Complementary Temporal Cycle-Consistent Adversarial Networks for Radar-Based Precipitation Nowcasting, in *32nd ACM International Conference on Information and Knowledge Management* 2023. [\[Paper\]](https://dl.acm.org/doi/abs/10.1145/3583780.3615006)
* A generative adversarial gated recurrent unit model for precipitation nowcasting, in *IEEE Geoscience and Remote Sensing Letters* 2019. [\[Paper\]](https://ieeexplore.ieee.org/abstract/document/8777193)
* Stochastic Super-Resolution for Downscaling Time-Evolving Atmospheric Fields With a Generative Adversarial Network, in *IEEE Transactions on Geoscience and Remote Sensing* 2020. [\[Paper\]](https://ieeexplore.ieee.org/abstract/document/9246532) [\[official code\]](https://github.com/jleinonen/downscaling-rnn-gan)
* Clgan: a generative adversarial network (gan)-based video prediction model for precipitation nowcasting, in *Geoscientific Model Development* 2023. [\[Paper\]](https://gmd.copernicus.org/articles/16/2737/2023/)
* Experimental study on generative adversarial network for precipitation nowcasting, in *IEEE Transactions on Geoscience and Remote Sensing* 2022. [\[Paper\]](https://ieeexplore.ieee.org/abstract/document/9780397)
* Skillful radar-based heavy rainfall nowcasting using task-segmented generative adversarial network, in *IEEE Transactions on Geoscience and Remote Sensing* 2023. [\[Paper\]](https://ieeexplore.ieee.org/abstract/document/10182305)
* A Generative Deep Learning Approach to Stochastic Downscaling of Precipitation Forecasts, in *Journal of Advances in Modeling Earth Systems* 2022. [\[Paper\]](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2022MS003120)
* Algorithmic Hallucinations of Near-Surface Winds: Statistical Downscaling with Generative Adversarial Networks to Convection-Permitting Scales, in *Artificial Intelligence for the Earth Systems* 2023. [\[Paper\]](https://journals.ametsoc.org/view/journals/aies/2/4/AIES-D-23-0015.1.xml)
* MSTCGAN: Multiscale Time Conditional Generative Adversarial Network for Long-Term Satellite Image Sequence Prediction, in *IEEE Transactions on Geoscience and Remote Sensing* 2022. [\[Paper\]](https://ieeexplore.ieee.org/abstract/document/9791392)
* Very Short-Term Rainfall Prediction Using Ground Radar Observations and Conditional Generative Adversarial Networks, in *IEEE Transactions on Geoscience and Remote Sensing* 2021. [\[Paper\]](https://ieeexplore.ieee.org/abstract/document/9532007)
* Physically constrained generative adversarial networks for improving precipitation fields from Earth system models, in *Nature Machine Intelligence* 2022. [\[Paper\]](https://www.nature.com/articles/s42256-022-00540-1)
* Producing realistic climate data with generative adversarial networks, in *Nonlinear Processes in Geophysics* 2021. [\[Paper\]](https://npg.copernicus.org/articles/28/347/2021/npg-28-347-2021-discussion.html) [\[official code\]](https://github.com/Cam-B04/Producing-realistic-climate-data-with-GANs)
* TemperatureGAN: Generative Modeling of Regional Atmospheric Temperatures, in *arXiv* 2023. [\[Paper\]](https://arxiv.org/abs/2306.17248)
* A Generative Adversarial Network for Climate Tipping Point Discovery (TIP-GAN), in *arXiv* 2023. [\[Paper\]](https://arxiv.org/abs/2302.10274)
* Physics-Guided Generative Adversarial Networks for Sea Subsurface Temperature Prediction, in *IEEE Transactions on Neural Networks and Learning Systems* 2021. [\[Paper\]](https://ieeexplore.ieee.org/abstract/document/9610615)
* Physical Knowledge-Enhanced Deep Neural Network for Sea Surface Temperature Prediction, in *IEEE Transactions on Geoscience and Remote Sensing* 2023. [\[Paper\]](https://ieeexplore.ieee.org/abstract/document/10068549)
* Physically-Consistent Generative Adversarial Networks for Coastal Flood Visualization, in  *arXiv* 2021. [\[Paper\]](https://arxiv.org/abs/2104.04785)
* A Space-Time Partial Differential Equation Based Physics-Guided Neural Network for Sea Surface Temperature Prediction, in *Remote Sensing* 2023. [\[Paper\]](https://www.mdpi.com/2072-4292/15/14/3498)
* Physics-informed generative neural network: an application to troposphere temperature prediction, in *Environmental Research Letters* 2021. [\[Paper\]](https://iopscience.iop.org/article/10.1088/1748-9326/abfde9/meta)

**Transformers-based Approaches**
* Oceanfourcast: Emulating Ocean Models with Transformers for Adjoint-based Data Assimilation, in *Copernicus Meetings* 2023. [\[Paper\]](https://meetingorganizer.copernicus.org/EGU23/EGU23-10810.html)
* Comprehensive Transformer-Based Model Architecture for Real-World Storm Prediction, in *Machine Learning and Knowledge Discovery in Databases* 2023. [\[Paper\]](https://link.springer.com/chapter/10.1007/978-3-031-43430-3_4)
* Transformer-based nowcasting of radar composites from satellite images for severe weather, in *arXiv* 2023. [\[Paper\]](https://arxiv.org/abs/2310.19515)
* Transformer for EI Niño-Southern Oscillation Prediction, in *IEEE Geoscience and Remote Sensing Letters* 2021. [\[Paper\]](https://ieeexplore.ieee.org/abstract/document/9504603)
* Spatiotemporal Swin-Transformer Network for Short Time Weather Forecasting, in *CIKM Workshops* 2021. [\[Paper\]](https://www.researchgate.net/profile/Hasan-Al-Marzouqi/publication/354371186_Spatiotemporal_Swin-Transformer_Network_for_Short_time_weather_forecasting/links/61c449a352bd3c7e05874c43/Spatiotemporal-Swin-Transformer-Network-for-Short-time-weather-forecasting.pdf)
* Towards physically consistent data-driven weather forecasting: Integrating data assimilation with equivariance-preserving deep spatial transformers, in *arXiv* 2021. [\[Paper\]](https://arxiv.org/abs/2103.09360)
* TENT: Tensorized Encoder Transformer for Temperature Forecasting, in *arXiv* 2021. [\[Paper\]](https://arxiv.org/abs/2106.14742) [\[official code\]](https://github.com/onurbil/TENT)
* A Novel Transformer Network With Shifted Window Cross-Attention for Spatiotemporal Weather Forecasting, in *IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing* 2023. [\[Paper\]](https://ieeexplore.ieee.org/abstract/document/10285372)
* Spatio-temporal interpretable neural network for solar irradiation prediction using transformer, in *Energy and Buildings* 2023. [\[Paper\]](https://www.sciencedirect.com/science/article/abs/pii/S0378778823006916)
* ClimaX: A foundation model for weather and climate, in *arXiv* 2023. [\[Paper\]](https://arxiv.org/abs/2301.10343) [\[official code\]](https://github.com/microsoft/ClimaX)
* Accurate medium-range global weather forecasting with 3D neural networks, in *Nature* 2023. [\[Paper\]](https://www.nature.com/articles/s41586-023-06185-3) [\[official code\]](https://github.com/198808xc/Pangu-Weather)
* W-MAE: Pre-trained weather model with masked autoencoder for multi-variable weather forecasting, in *arXiv* 2023. [\[Paper\]](https://arxiv.org/abs/2304.08754) [\[official code\]](https://github.com/gufrannn/w-mae)
* FengWu: Pushing the Skillful Global Medium-range Weather Forecast beyond 10 Days Lead, in *arXiv* 2023. [\[Paper\]](https://arxiv.org/abs/2304.02948)
* Improving medium-range ensemble weather forecasts with hierarchical ensemble transformers, in *arXiv* 2023. [\[Paper\]](https://arxiv.org/abs/2303.17195)
* CliMedBERT: A Pre-trained Language Model for Climate and Health-related Text, in *arXiv* 2022. [\[Paper\]](https://arxiv.org/abs/2212.00689)
* ClimateBERT-NetZero: Detecting and Assessing Net Zero and Reduction Targets, in *arXiv* 2023. [\[Paper\]](https://arxiv.org/abs/2310.08096)
* Fine-tuning ClimateBert transformer with ClimaText for the disclosure analysis of climate-related financial risks, in *arXiv* 2023. [\[Paper\]](https://arxiv.org/abs/2303.13373)
* ChatClimate: Grounding Conversational AI in Climate Science, in *arXiv* 2023. [\[Paper\]](https://www.researchgate.net/profile/Jingwei-Ni/publication/369975012_chatIPCC_Grounding_Conversational_AI_in_Climate_Science/links/645175364af7887352518782/chatIPCC-Grounding-Conversational-AI-in-Climate-Science.pdf)
* ClimateNLP: Analyzing Public Sentiment Towards Climate Change Using Natural Language Processing, in *arXiv* 2023. [\[Paper\]](https://arxiv.org/abs/2310.08099)
* Evaluating TCFD Reporting: A New Application of Zero-Shot Analysis to Climate-Related Financial Disclosures, in *arXiv* 2023. [\[Paper\]](https://arxiv.org/abs/2302.00326)
* Enhancing Large Language Models with Climate Resources, in *arXiv* 2023. [\[Paper\]](https://arxiv.org/abs/2304.00116)

**Graph Neural Networks-based Approaches**
* ENSO-GTC: ENSO Deep Learning Forecast Model With a Global Spatial-Temporal Teleconnection Coupler, in *Journal of Advances in Modeling Earth Systems* 2022. [\[Paper\]](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2022MS003132) [\[official code\]](https://github.com/BrunoQin/ENSO-GTC)
* GraphCast: Learning skillful medium-range global weather forecasting, in *arXiv* 2022. [\[Paper\]](https://arxiv.org/abs/2212.12794) [\[official code\]](https://github.com/google-deepmind/graphcast)
* Forecasting Global Weather with Graph Neural Networks, in *arXiv* 2022. [\[Paper\]](https://arxiv.org/abs/2202.07575) [\[official code\]](https://github.com/openclimatefix/graph_weather)
* GE-STDGN: a novel spatio-temporal weather prediction model based on graph evolution, in *Applied Intelligence* 2022. [\[Paper\]](https://link.springer.com/article/10.1007/s10489-021-02824-2) [\[official code\]](https://github.com/fatekong/GE-STDGN)
* HiSTGNN: Hierarchical spatio-temporal graph neural network for weather forecasting, in *Information Sciences* 2023. [\[Paper\]](https://www.sciencedirect.com/science/article/abs/pii/S0020025523011659)
* Convolutional GRU Network for Seasonal Prediction of the El Niño-Southern Oscillation, in *arXiv* 2023. [\[Paper\]](https://arxiv.org/abs/2306.10443)
* DK-STN: A Domain Knowledge Embedded Spatio-Temporal Network Model for MJO Forecast, in *Expert Systems With Applications, Forthcoming* 2023. [\[Paper\]](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4574792)
* ClimART: A Benchmark Dataset for Emulating Atmospheric Radiative Transfer in Weather and Climate Models, in *arXiv* 2021. [\[Paper\]](https://arxiv.org/abs/2111.14671) [\[official code\]](https://github.com/RolnickLab/climart)
* A Low Rank Weighted Graph Convolutional Approach to Weather Prediction, in *IEEE International Conference on Data Mining (ICDM)* 2018. [\[Paper\]](https://ieeexplore.ieee.org/abstract/document/8594887) [\[official code\]](https://github.com/TylerPWilson/wgc-lstm)
* WeKG-MF: A Knowledge Graph of Observational Weather Data, in *European Semantic Web Conference* 2022. [\[Paper\]](https://link.springer.com/chapter/10.1007/978-3-031-11609-4_19)
* Regional Heatwave Prediction Using Graph Neural Network and Weather Station Data, in *Geophysical Research Letters* 2023. [\[Paper\]](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2023GL103405)
* Graph-based Neural Weather Prediction for Limited Area Modeling, in *arXiv* 2023. [\[Paper\]](https://arxiv.org/abs/2309.17370) [\[official code\]](https://github.com/joeloskarsson/neural-lam)
* Joint Air Quality and Weather Prediction Based on Multi-Adversarial Spatiotemporal Networks, in *AAAI* 2021. [\[Paper\]](https://ojs.aaai.org/index.php/AAAI/article/view/16529)
* Semi-Supervised Air Quality Forecasting via Self-Supervised Hierarchical Graph Neural Network, in *IEEE Transactions on Knowledge and Data Engineering* 2022. [\[Paper\]](https://ieeexplore.ieee.org/abstract/document/9709128)
* CNGAT: A Graph Neural Network Model for Radar Quantitative Precipitation Estimation, in *IEEE Transactions on Geoscience and Remote Sensing* 2021. [\[Paper\]](https://ieeexplore.ieee.org/abstract/document/9570292)
* Prompt Federated Learning for Weather Forecasting: Toward Foundation Models on Meteorological Data, in *arXiv* 2023. [\[Paper\]](https://arxiv.org/abs/2301.09152) [\[official code\]](https://github.com/shengchaochen82/MetePFL)
* Spatial-temporal Prompt Learning for Federated Weather Forecasting, in *arXiv* 2023. [\[Paper\]](https://arxiv.org/abs/2305.14244)

---
  
## Application

**Forecasting**
* Dwfh: An improved data-driven deep weather forecasting hybrid model using transductive long short term memory (t-lstm), in *EAAI* 2023. [\[Paper\]](https://www.sciencedirect.com/science/article/pii/S0957417422022886)
* Swinrdm: integrate swinrnn with diffusion model towards high-resolution and highquality weather forecasting, in *AAAI* 2023. [\[Paper\]](https://ojs.aaai.org/index.php/AAAI/article/view/25105)
* Swinvrnn: A data-driven ensemble forecasting model via learned distribution perturbation, in *Journal of Advances in Modeling Earth Systems* 2023. [\[Paper\]](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2022MS003211)
* Time-series prediction of hourly atmospheric pressure using anfis and lstm approaches, in *Neural Computing and Applications* 2022. [\[Paper\]](https://link.springer.com/article/10.1007/s00521-022-07275-5) 
* Ilf-lstm: Enhanced loss function in lstm to predict the sea surface temperature, in *Soft Computing* 2022. [\[Paper\]](https://link.springer.com/article/10.1007/s00500-022-06899-y) 
* FourCastNet: A Global Data-driven High-resolution Weather Model using Adaptive Fourier Neural Operator, in *arXiv* 2022. [\[paper\]](https://arxiv.org/abs/2202.11214) [\[official code\]](https://github.com/NVlabs/FourCastNet)
* An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale, in *arXiv* 2020. [\[Paper\]](https://arxiv.org/abs/2010.11929) [\[official code\]](https://github.com/gupta-abhay/pytorch-vit)
* Improving medium-range ensemble weather forecasts with hierarchical ensemble transformers, in *arXiv* 2023. [\[Paper\]](https://arxiv.org/abs/2303.17195)
* TeleViT: Teleconnection-Driven Transformers Improve Subseasonal to Seasonal Wildfire Forecasting, in *ICCV* 2023. [\[Paper\]](https://openaccess.thecvf.com/content/ICCV2023W/AIHADR/html/Prapas_TeleViT_Teleconnection-Driven_Transformers_Improve_Subseasonal_to_Seasonal_Wildfire_Forecasting_ICCVW_2023_paper.html) [\[official code\]](https://github.com/Orion-AI-Lab/televit)
* Accurate Medium-Range Global Weather Forecasting with 3D Neural Networks, in *Nature* 2023. [\[paper\]](https://www.nature.com/articles/s41586-023-06185-3) [\[official code\]](https://github.com/198808xc/Pangu-Weather)
* FengWu: Pushing the Skillful Global Medium-range Weather Forecast beyond 10 Days Lead, in *arXiv* 2023. [\[paper\]](https://arxiv.org/abs/2304.02948)
* FuXi: A cascade machine learning forecasting system for 15-day global weather forecast, in *arXiv* 2023. [\[paper\]](https://arxiv.org/abs/2306.12873) [\[official code\]](https://github.com/tpys/FuXi)
* FuXi-Extreme: Improving extreme rainfall and wind forecasts with diffusion model, in *arXiv* 2023. [\[paper\]](https://arxiv.org/abs/2310.19822)
* Denoising Diffusion Probabilistic Models, in *NeurIPS* 2020. [\[paper\]](https://proceedings.neurips.cc/paper/2020/hash/4c5bcfec8584af0d967f1ab10179ca4b-Abstract.html) [\[official code\]](https://github.com/hojonathanho/diffusion)
* ClimaX: A Foundation Model for Weather and Climate, in *arXiv* 2023. [\[paper\]](https://arxiv.org/abs/2301.10343) [\[official code\]](https://github.com/microsoft/ClimaX)
* W-MAE: Pre-Trained Weather Model with Masked Autoencoder for Multi-Variable Weather Forecasting, in *arXiv* 2023. [\[paper\]](https://arxiv.org/abs/2304.08754)  [\[official code\]](https://github.com/Gufrannn/W-MAE)
* Masked Autoencoders Are Scalable Vision Learners, in *CVPR* 2022. [\[paper\]](https://openaccess.thecvf.com/content/CVPR2022/html/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022_paper)  [\[official code\]](https://github.com/pengzhiliang/MAE-pytorch)
* Masked Autoencoders As Spatiotemporal Learners, in *NeurIPS* 2022. [\[paper\]](https://proceedings.neurips.cc/paper_files/paper/2022/hash/e97d1081481a4017df96b51be31001d3-Abstract-Conference.html) [\[official code\]](https://github.com/facebookresearch/mae_st)
* SEEDS: Emulation of Weather Forecast Ensembles with Diffusion Models, in *arXiv* 2023. [\[Paper\]](https://arxiv.org/abs/2306.14066)
* DYffusion: A Dynamics-informed Diffusion Model for Spatiotemporal Forecasting, in *arXiv* 2023. [\[Paper\]](https://arxiv.org/abs/2306.01984) [\[official code\]](https://github.com/Rose-STL-Lab/dyffusion)
* PDE-Refiner: Achieving Accurate Long Rollouts with Neural PDE Solvers, in *arXiv* 2023. [\[Paper\]](https://arxiv.org/abs/2308.05732)
* DiTTO: Diffusion-inspired Temporal Transformer Operator, in *arXiv* 2023. [\[Paper\]](https://arxiv.org/abs/2307.09072)
* TemperatureGAN: Generative Modeling of Regional Atmospheric Temperatures, in *arXiv* 2023. [\[Paper\]](https://arxiv.org/abs/2306.17248)
* Physics-Guided Generative Adversarial Networks for Sea Subsurface Temperature Prediction, in *IEEE Transactions on Neural Networks and Learning Systems* 2021. [\[Paper\]](https://ieeexplore.ieee.org/abstract/document/9610615)
* Physical Knowledge-Enhanced Deep Neural Network for Sea Surface Temperature Prediction, in *IEEE Transactions on Geoscience and Remote Sensing* 2023. [\[Paper\]](https://ieeexplore.ieee.org/abstract/document/10068549)
* Physically-Consistent Generative Adversarial Networks for Coastal Flood Visualization, in  *arXiv* 2021. [\[Paper\]](https://arxiv.org/abs/2104.04785)
* A Space-Time Partial Differential Equation Based Physics-Guided Neural Network for Sea Surface Temperature Prediction, in *Remote Sensing* 2023. [\[Paper\]](https://www.mdpi.com/2072-4292/15/14/3498)
* Physics-informed generative neural network: an application to troposphere temperature prediction, in *Environmental Research Letters* 2021. [\[Paper\]](https://iopscience.iop.org/article/10.1088/1748-9326/abfde9/meta)
* A Low Rank Weighted Graph Convolutional Approach to Weather Prediction, in *IEEE International Conference on Data Mining (ICDM)* 2018. [\[Paper\]](https://ieeexplore.ieee.org/abstract/document/8594887) [\[official code\]](https://github.com/TylerPWilson/wgc-lstm)
* Forecasting Global Weather with Graph Neural Networks, in *arXiv* 2022. [\[Paper\]](https://arxiv.org/abs/2202.07575) [\[official code\]](https://github.com/openclimatefix/graph_weather)
* The Graph Neural Network Model, in *IEEE Transactions on Neural Networks* 2008. [\[official code\]](https://ieeexplore.ieee.org/abstract/document/4700287) [\[official code\]](https://github.com/pyg-team/pytorch_geometric)
* GraphCast: Learning Skillful Medium-Range Global Weather Forecasting, in *arXiv* 2022. [\[paper\]](https://arxiv.org/abs/2212.12794) [\[official code\]](https://github.com/google-deepmind/graphcast)
* The World as a Graph: Improving El Niño Forecasts with Graph Neural Networks, in *arXiv* 2021. [\[paper\]](https://arxiv.org/abs/2104.05089) [\[official code\]](https://github.com/salvaRC/Graphino)
* ENSO analysis and prediction using deep learning: A review, in *Neurocomputing* 2023. [\[paper\]](https://www.sciencedirect.com/science/article/pii/S0925231222014722)
* GE-STDGN: a novel spatio-temporal weather prediction model based on graph evolution, in *Applied Intelligence* 2022. [\[Paper\]](https://link.springer.com/article/10.1007/s10489-021-02824-2) [\[official code\]](https://github.com/fatekong/GE-STDGN)
* Graph evolution: Densification and shrinking diameters, in *TKDD* 2007. [\[Paper\]](https://dl.acm.org/doi/abs/10.1145/1217299.1217301) [\[official code\]](https://github.com/dhivyaeswaran/hols)
* HiSTGNN: Hierarchical spatio-temporal graph neural network for weather forecasting, in *Information Sciences* 2023. [\[Paper\]](https://www.sciencedirect.com/science/article/abs/pii/S0020025523011659)
* Hierarchical Graph Representation Learning with Differentiable Pooling, in *NeurIPS* 2018. [\[Paper\]](https://proceedings.neurips.cc/paper_files/paper/2018/hash/e77dbaf6759253c7c6d0efc5690369c7-Abstract.html) [\[official code\]](https://github.com/murphyyhuang/gnn_hierarchical_pooling)
* WeKG-MF: A Knowledge Graph of Observational Weather Data, in *European Semantic Web Conference* 2022. [\[Paper\]](https://link.springer.com/chapter/10.1007/978-3-031-11609-4_19)

**Precipitation Nowcasting**

* Dynamic Multiscale Fusion Generative Adversarial Network for Radar Image Extrapolation, in *IEEE Transactions on Geoscience and Remote Sensing* 2022. [\[Paper\]](https://ieeexplore.ieee.org/abstract/document/9837952)
* MCSIP Net: Multichannel Satellite Image Prediction via Deep Neural Network, in *IEEE Transactions on Geoscience and Remote Sensing* 2019. [\[Paper\]](https://ieeexplore.ieee.org/abstract/document/8933126)
* Developing Deep Learning Models for Storm Nowcasting, in *IEEE Transactions on Geoscience and Remote Sensing* 2021. [\[Paper\]](https://ieeexplore.ieee.org/abstract/document/9542945)
* Enhancing Spatial Variability Representation of Radar Nowcasting with Generative Adversarial Networks, in *Remote Sensing* 2023. [\[Paper\]](https://www.mdpi.com/2072-4292/15/13/3306) [\[official code\]](https://github.com/THUGAF/SVRE-Nowcasting)
* NowCasting-Nets: Representation Learning to Mitigate Latency Gap of Satellite Precipitation Products Using Convolutional and Recurrent Neural Networks, in *IEEE Transactions on Geoscience and Remote Sensing* 2022. [\[Paper\]](https://ieeexplore.ieee.org/abstract/document/9732949) [\[official code\]](https://github.com/rehsani/NowCasting-nets)
* Broad-UNet: Multi-scale feature learning for nowcasting tasks, in *Neural Networks* 2021. [\[Paper\]]([\[Paper\]](https://www.sciencedirect.com/science/article/pii/S089360802100349X)) [\[official code\]](https://github.com/jesusgf96/Broad-UNet)
* Dynamic Multiscale Fusion Generative Adversarial Network for Radar Image Extrapolation, in *IEEE Transactions on Geoscience and Remote Sensing* 2022. [\[Paper\]](https://ieeexplore.ieee.org/abstract/document/9837952)
* Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting, in *NeurIPS* 2015. [\[Paper\]](https://proceedings.neurips.cc/paper/2015/hash/07563a3fe3bbe7e3ba84431ad9d055af-Abstract.html) [\[official code\]](https://github.com/ndrplz/ConvLSTM_pytorch)
* MSTCGAN: Multiscale Time Conditional Generative Adversarial Network for Long-Term Satellite Image Sequence Prediction, in *IEEE Transactions on Geoscience and Remote Sensing* 2022. [\[Paper\]](https://ieeexplore.ieee.org/abstract/document/9791392) 
* MMSTN: A Multi-Modal Spatial-Temporal Network for Tropical Cyclone Short-Term Prediction, in *Geophysical Research Letters* 2022. [\[Paper\]](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2021GL096898)
* PFST-LSTM: A SpatioTemporal LSTM Model With Pseudoflow Prediction for Precipitation Nowcasting, in *IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing* 2020. [\[official code\]](https://github.com/luochuyao/PFST-LSTM)
* TempEE: Temporal-Spatial Parallel Transformer for Radar Echo Extrapolation Beyond Auto-Regression, in *arXiv* 2023. [\[Paper\]](https://arxiv.org/abs/2304.14131)
* Nowformer : A Locally Enhanced Temporal Learner for Precipitation Nowcasting. [\[Paper\]](https://s3.us-east-1.amazonaws.com/climate-change-ai/papers/neurips2022/80/paper.pdf)
* Rainformer: Features Extraction Balanced Network for Radar-Based Precipitation Nowcasting, in *IEEE Geoscience and Remote Sensing Letters* 2022. [\[Paper\]](https://ieeexplore.ieee.org/abstract/document/9743916) [\[official code\]](https://github.com/Zjut-MultimediaPlus/Rainformer)
* PTCT: Patches with 3D-Temporal Convolutional Transformer Network for Precipitation Nowcasting, in *arXiv* 2021. [\[Paper\]](https://arxiv.org/abs/2112.01085) [\[official code\]](https://github.com/yangziao56/TCTN-pytorch)
* Preformer: Simple and Efficient Design for Precipitation Nowcasting with Transformers, in *IEEE Geoscience and Remote Sensing Letters* 2023. [\[Paper\]](https://ieeexplore.ieee.org/abstract/document/10288072)
* Motion-Guided Global–Local Aggregation Transformer Network for Precipitation Nowcasting, in *IEEE Transactions on Geoscience and Remote Sensing* 2022. [\[Paper\]](https://ieeexplore.ieee.org/abstract/document/9931154)
* Predrnn: A recurrent neural network for spatiotemporal predictive learning, in *IEEE T-PAMI* 2022. [\[Paper\]](https://ieeexplore.ieee.org/abstract/document/9749915) [\[official code\]](https://github.com/thuml/predrnn-pytorch) 
* Eidetic 3d lstm: A model for video prediction and beyond, in *ICLR* 2018. [\[Paper\]](http://faculty.ucmerced.edu/mhyang/papers/iclr2019_eidetic3d.pdf) [\[official code\]](https://github.com/google/e3d_lstm) 
* Disentangling Physical Dynamics From Unknown Factors for Unsupervised Video Prediction, in *CVPR* 2020. [\[Paper\]](https://openaccess.thecvf.com/content_CVPR_2020/html/Le_Guen_Disentangling_Physical_Dynamics_From_Unknown_Factors_for_Unsupervised_Video_Prediction_CVPR_2020_paper.html)
* Partial differential equations, in *American Mathematical Society* 2022. [\[Paper\]](https://books.google.co.jp/books?hl=zh-CN&lr=&id=Ott1EAAAQBAJ&oi=fnd&pg=PP1&dq=Partial+differential+equations&ots=cVEvwI4QvJ&sig=Y1ulehDMpv87Eddv8_cdvx7hJug&redir_esc=y#v=onepage&q=Partial%20differential%20equations&f=false)
* Metnet: A neural weather model for precipitation forecasting, in *arXiv* 2020. [\[Paper\]](https://arxiv.org/abs/2003.12140) [\[official code\]](https://github.com/openclimatefix/metnet)
* Deep Learning for Day Forecasts from Sparse Observations, in *arXiv* 2023. [\[Paper\]](https://arxiv.org/abs/2306.06079)
* Earthformer: Exploring Space-Time Transformers for Earth System Forecasting, in *NeurIPS* 2022. [\[Paper\]](https://proceedings.neurips.cc/paper_files/paper/2022/hash/a2affd71d15e8fedffe18d0219f4837a-Abstract-Conference.html) [\[official code\]](https://github.com/amazon-science/earth-forecasting-transformer)
* ENSO and greenhouse warming, in *Nature Climate Change* 2015. [\[Paper\]](https://www.nature.com/articles/nclimate2743)
* MM-RNN: A Multimodal RNN for Precipitation Nowcasting, in *IEEE Transactions on Geoscience and Remote Sensing* 2023. [\[Paper\]](https://ieeexplore.ieee.org/abstract/document/10092888)
* Spatiotemporal inference network for precipitation nowcasting with multi-modal fusion, in *IEEE Journal of Selected Topics in Applied Earth Observation and Remote Sensing* 2023.[\[Paper\]](https://ieeexplore.ieee.org/abstract/document/10285341)
* PCT-CycleGAN: Paired Complementary Temporal Cycle-Consistent Adversarial Networks for Radar-Based Precipitation Nowcasting, in *32nd ACM International Conference on Information and Knowledge Management* 2023. [\[Paper\]](https://dl.acm.org/doi/abs/10.1145/3583780.3615006)
* MPL-GAN: Toward Realistic Meteorological Predictive Learning Using Conditional GAN, in *IEEE Access* 2020. [\[Paper\]](https://ieeexplore.ieee.org/abstract/document/9094665)
* PreDiff: Precipitation Nowcasting with Latent Diffusion Models, in *arXiv* 2023. [\[Paper\]](https://arxiv.org/abs/2307.10422)
* Precipitation nowcasting with generative diffusion models, in *arXiv* 2023. [\[Paper\]](https://arxiv.org/abs/2308.06733)
* Skilful precipitation nowcasting using deep generative models of radar, in *Nature* 2021. [\[Paper\]](https://www.nature.com/articles/s41586-021-03854-z) [\[official code\]](https://github.com/openclimatefix/skillful_nowcasting)
* Physically constrained generative adversarial networks for improving precipitation fields from Earth system models, in *Nature Machine Intelligence* 2022. [\[Paper\]](https://www.nature.com/articles/s42256-022-00540-1)
* CNGAT: A Graph Neural Network Model for Radar Quantitative Precipitation Estimation, in *IEEE Transactions on Geoscience and Remote Sensing* 2021. [\[Paper\]](https://ieeexplore.ieee.org/abstract/document/9570292)
* Multi-Radar Multi-Sensor (MRMS) Quantitative Precipitation Estimation: Initial Operating Capabilities, in *Bulletin of the American Meteorological Society* 2016. [\[Paper\]](https://journals.ametsoc.org/view/journals/bams/97/4/bams-d-14-00174.1.xml)

---
  
## Dataset

**Weather and Climate Series Data**
* WeatherBench: A Benchmark Data Set for Data-Driven Weather Forecasting, in *Journal of Advances in Modeling Earth Systems* 2020. [\[Paper\]](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2020MS002203) [\[official code\]](https://github.com/pangeo-data/WeatherBench)
* WeatherBench 2: A benchmark for the next generation of data-driven global weather models, in *arXiv* 2023. [\[Paper\]](https://arxiv.org/abs/2308.15560) [\[official code\]](https://github.com/google-research/weatherbench2)
* ClimateLearn: Benchmarking Machine Learning for Weather and Climate Modeling, in *arXiv* 2023. [\[Paper\]](https://arxiv.org/abs/2307.01909) [\[official code\]](https://github.com/aditya-grover/climate-learn)
* An Evaluation and Intercomparison of Global Analyses from the National Meteorological Center and the European Centre for Medium Range Weather Forecasts, in *Bulletin of the American Meteorological Society* 1988. [\[Paper\]](https://journals.ametsoc.org/view/journals/bams/69/9/1520-0477_1988_069_1047_aeaiog_2_0_co_2.xml)
* SODA: A Reanalysis of Ocean Climate, in *Journal of Geophysical Research-Oceans* 2005. [\[Paper\]](https://www2.atmos.umd.edu/~carton/pdfs/carton&giese05.pdf)
* DroughtED: A dataset and methodology for drought forecasting spanning multiple climate zones, in *ICML* 2021. [\[Paper\]](https://s3.us-east-1.amazonaws.com/climate-change-ai/papers/icml2021/22/paper.pdf)
* Digital Typhoon: Long-term Satellite Image Dataset for the Spatio-Temporal Modeling of Tropical Cyclones, in *arXiv* 2023. [\[Paper\]](https://arxiv.org/abs/2311.02665) [\[official code\]](https://github.com/kitamoto-lab/digital-typhoon)
* EarthNet2021: A Large-Scale Dataset and Challenge for Earth Surface Forecasting as a Guided Video Prediction Task, in *Computer Vision and Pattern Recognition* 2021. [\[Paper\]](https://openaccess.thecvf.com/content/CVPR2021W/EarthVision/html/Requena-Mesa_EarthNet2021_A_Large-Scale_Dataset_and_Challenge_for_Earth_Surface_Forecasting_CVPRW_2021_paper.html) [\[official code\]](https://github.com/earthnet2021/earthnet-model-intercomparison-suite)
* ClimateNet: an expert-labeled open dataset and deep learning architecture for enabling high-precision analyses of extreme weather, in *Geoscientific Model Development* 2021. [\[Paper\]](https://gmd.copernicus.org/articles/14/107/2021/) [\[official code\]](https://mega.nz/file/pDMAAajR#GvUg7JV_HmByDLJYS1w6mEw9nh9o9f_YM_v9jl1R1Cw)
* IowaRain: A Statewide Rain Event Dataset Based on Weather Radars and Quantitative Precipitation Estimation, in *arXiv* 2021. [\[Paper\]](https://arxiv.org/abs/2107.03432) [\[official code\]](https://github.com/uihilab/IowaRain)
* ExtremeWeather: A large-scale climate dataset for semi-supervised detection, localization, and understanding of extreme weather events, in *NeurIPS* 2017. [\[Paper\]](https://proceedings.neurips.cc/paper_files/paper/2017/hash/519c84155964659375821f7ca576f095-Abstract.html) [\[official code\]](https://github.com/eracah/hur-detect)
* Benchmark Dataset for Precipitation Forecasting by Post-Processing the Numerical Weather Prediction, in *arXiv* 2022. [\[Paper\]](https://arxiv.org/abs/2206.15241) [\[official code\]](https://github.com/osilab-kaist/komet-benchmark-dataset)
* A gridded dataset of hourly precipitation in Germany: Its construction, climatology and application, in *Meteorologische Zeitschrift* 2008. [\[Paper\]](https://elib.dlr.de/57270/)
* PostRainBench: A comprehensive benchmark and a new model for precipitation forecasting, in *arXiv* 2023. [\[Paper\]](https://arxiv.org/abs/2310.02676)
* 1 km monthly temperature and precipitation dataset for China from 1901 to 2017, in *Earth System Science Data* 2019. [\[Paper\]](https://essd.copernicus.org/articles/11/1931/2019/)
* ClimART: A Benchmark Dataset for Emulating Atmospheric Radiative Transfer in Weather and Climate Models, in *arXiv* 2021. [\[Paper\]](https://arxiv.org/abs/2111.14671) [\[official code\]](https://github.com/RolnickLab/climart)
* Rain-F: A Fusion Dataset for Rainfall Prediction Using Convolutional Neural Network, in *IGARSS* 2021. [\[Paper\]](https://ieeexplore.ieee.org/abstract/document/9555094)
* RAIN-F+: The Data-Driven Precipitation Prediction Model for Integrated Weather Observations, in *Remote Sensing* 2021. [\[Paper\]](https://www.mdpi.com/2072-4292/13/18/3627) [\[official code\]](https://github.com/chagmgang/cv)
* ENS-10: A Dataset For Post-Processing Ensemble Weather Forecasts, in *NeurIPS* 2022. [\[Paper\]](https://proceedings.neurips.cc/paper_files/paper/2022/hash/89e44582fd28ddfea1ea4dcb0ebbf4b0-Abstract-Datasets_and_Benchmarks.html) [\[official code\]](https://github.com/spcl/ens10)
* SEVIR : A Storm Event Imagery Dataset for Deep Learning Applications in Radar and Satellite Meteorology, in *NeurIPS* 2020. [\[Paper\]](https://proceedings.neurips.cc/paper/2020/hash/fa78a16157fed00d7a80515818432169-Abstract.html) [\[official code\]](https://github.com/MIT-AI-Accelerator/neurips-2020-sevir)
* RainBench: Towards Data-Driven Global Precipitation Forecasting from Satellite Imagery, in *AAAI* 2021. [\[Paper\]](https://ojs.aaai.org/index.php/AAAI/article/view/17749) [\[official code\]](https://github.com/frontierdevelopmentlab/pyrain)
* PM2.5-GNN: A Domain Knowledge Enhanced Graph Neural Network For PM2.5 Forecasting, in *Advances in Geographic Information Systems* 2020. [\[Paper\]](https://dl.acm.org/doi/abs/10.1145/3397536.3422208) [\[official code\]](https://github.com/shuowang-ai/PM2.5-GNN)
* Weather2K: A Multivariate Spatio-Temporal Benchmark Dataset for Meteorological Forecasting Based on Real-Time Observation Data from Ground Weather Stations, in *arXiv* 2023. [\[Paper\]](https://arxiv.org/abs/2302.10493) [\[official code\]](https://github.com/bycnfz/weather2k/)
* Prompt Federated Learning for Weather Forecasting: Toward Foundation Models on Meteorological Data, in *arXiv* 2023. [\[Paper\]](https://arxiv.org/abs/2301.09152) [\[official code\]](https://github.com/shengchaochen82/MetePFL)
* LSDSSIMR: Large-Scale Dust Storm Database Based on Satellite Images and Meteorological Reanalysis Data, in *IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing* 2023. [\[Paper\]](https://ieeexplore.ieee.org/abstract/document/10287393) [\[official code\]](https://github.com/Zjut-MultimediaPlus/LSDSSIMR)

**Weather and Climate Text Data**
* CLIMATE-FEVER: A Dataset for Verification of Real-World Climate Claims, in *arXiv* 2021. [\[Paper\]](https://arxiv.org/abs/2012.00614) [\[official code\]](https://github.com/tdiggelm/climate-fever-dataset)
* ClimateBERT-NetZero: Detecting and Assessing Net Zero and Reduction Targets, in *arXiv* 2023. [\[Paper\]](https://arxiv.org/abs/2310.08096)
* ClimaText: A Dataset for Climate Change Topic Detection, in *arXiv* 2020. [\[Paper\]](https://arxiv.org/abs/2012.00483)
* Towards Fine-grained Classification of Climate Change related Social Media Text, in *Association for Computational Linguistics: Student Research Workshop* 2022. [\[Paper\]](https://aclanthology.org/2022.acl-srw.35/)
* Neuralnere: Neural named entity relationship extraction for end-to-end climate change knowledge graph construction, in *ICML* 2021. [\[Paper\]](https://s3.us-east-1.amazonaws.com/climate-change-ai/papers/icml2021/76/paper.pdf)


## Please cite our publication if you found our research to be helpful.

```bibtex
@article{chen2023foundation,
  title={Foundation models for weather and climate data understanding: A comprehensive survey},
  author={Chen, Shengchao and Long, Guodong and Jiang, Jing and Liu, Dikai and Zhang, Chengqi},
  journal={arXiv preprint arXiv:2312.03014},
  year={2023}
}
