<div align="center">
<img src="docs/sjtutes2.png" alt="logo" width="750"/>
</div>

**SJTU-TES**, SJTU Technology Engage Square, is an inclusive platform that aims to replicate cutting-edge technologies from diverse fields, enabling beginners to gain practical experience through hands-on projects and activities. 

We mark work contributed by **SJTU** with ⭐.

## [Content](#content)

<table>
<tr><td colspan="2"><a href="#AIGC">1. AIGC</a></td></tr> 
<tr>
	<td>&emsp;<a href=#text2img>1.1 Text2Img</a></td>
	<td>&emsp;<a href=#text2video>1.2 Text2Video</a></td>
</tr>
<tr>
	<td>&emsp;<a href=#deepfake>1.3 DeepFake</a></td>
</tr>


<tr><td colspan="2"><a href="#CO">2. CO</a></td></tr> 
<tr>
	<td>&emsp;<a href=#graph-matching>2.1 Graph Matching (GM)</a></td>
	<td>&emsp;<a href=#graph-edit-distance>2.2 Graph Edit Distance (GED)</a></td>
</tr>
<tr>
	<td>&emsp;<a href=#travelling-salesman-problem>2.3 Travelling Salesman Problem (TSP)</a></td>
	<td>&emsp;<a href=#maximum-independent-set>2.4 Maximum Independent Set (MIS)</a></td>
</tr>


<tr><td colspan="2"><a href="#website">3. Website</a></td></tr> 
<tr>
	<td>&emsp;<a href=#online-chatting>3.1 Online Chatting</a></td>
	<td>&emsp;<a href=#web-scraping>3.2 Web Scraping</a></td>
</tr>
<tr>
	<td>&emsp;<a href=#cpp-online>3.3 Cpp Online</a></td>
</tr>

<tr><td colspan="2"><a href="#motion">4. Motion</a></td></tr> 
<tr>
	<td>&emsp;<a href=#motion-retargeting>4.1 Motion Retargeting</a></td>
	<td>&emsp;<a href=#pose-estimation>4.2 Pose Estimation</a></td>
</tr>
<tr>
	<td>&emsp;<a href=#video-matting>4.3 Video Matting</a></td>
</tr>

<tr><td colspan="2"><a href="#security">5. Security</a></td></tr> 
<tr>
	<td>&emsp;<a href=#deepLearning-security>5.1 DeepLearning Security</a></td>
	<td>&emsp;<a href=#iot-security>5.2 IOT Security</a></td>
</tr>

<tr><td colspan="2"><a href="#tool">6. Tool</a></td></tr> 
<tr>
	<td>&emsp;<a href=#clearAgenda-scheduler>6.1 ClearAgenda Scheduler</a></td>
</tr>

</table>

## [AIGC](#content)

### [Text2Img](#content)

#### 1.1.1 Stable Diffusion v1.4 [![GitHub stars](https://img.shields.io/github/stars/CompVis/stable-diffusion?style=social&label=Star&maxAge=8640)](https://GitHub.com/CompVis/stable-diffusion/) 

``Stable Diffusion``, **a latent text-to-image diffusion model capable of generating photo-realistic images given any text input.** ``stable-diffusion-v1-4`` is resumed from ``stable-diffusion-v1-2`` - 225,000 steps at resolution 512x512 on ``laion-aesthetics v2 5+`` and 10 % dropping of the text-conditioning to improve.
[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.pdf), [code](https://github.com/CompVis/stable-diffusion), [pretrained](https://huggingface.co/CompVis/stable-diffusion-v-1-4-original)

#### 1.1.2 Stable Diffusion v1.5 [![GitHub stars](https://img.shields.io/github/stars/CompVis/stable-diffusion?style=social&label=Star&maxAge=8640)](https://GitHub.com/CompVis/stable-diffusion/) 

The ``stable-diffusion-v1-5`` checkpoint was initialized with the weights of the ``stable-diffusion-v1-2`` checkpoint and subsequently fine-tuned on 595k steps at resolution 512x512 on ``laion-aesthetics v2 5+`` and 10% dropping of the text-conditioning to improve classifier-free guidance sampling. [paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.pdf), [code](https://github.com/CompVis/stable-diffusion), [pretrained](https://huggingface.co/runwayml/stable-diffusion-v1-5), [repro](https://github.com/SJTU-TES/repro-stable-diffusion)
<details>
<summary>Click to view examples we have implemented</summary>

- Scarlett, nature, (((beauty))), (((smooth)))，white，Highest quality

<img src="docs/examples/sd_scarlett.png" width="80%" alt="" />
</details>

### [Text2Video](#content)

#### 1.2.1 Latte [![GitHub stars](https://img.shields.io/github/stars/Vchitect/Latte?style=social&label=Star&maxAge=8640)](https://GitHub.com/Vchitect/Latte/) 

``Latte``, **a novel latent diffusion transformer for video generation**, utilizes spatio-temporal tokens extracted from input videos and employs a series of Transformer blocks to model the distribution of videos in the latent space. Latte achieves state-of-the-art performance on four standard video generation datasets ``FaceForensics``, ``SkyTimelapse``, ``UCF101``, and ``Taichi-HD``.

[paper](https://arxiv.org/pdf/2401.03048v1.pdf), [code](https://github.com/Vchitect/Latte?tab=readme-ov-file), [pretrained](https://huggingface.co/maxin-cn/Latte), [repro](https://github.com/SJTU-TES/repro-latte)

<details>
<summary>Click to view examples we have implemented</summary>

- Yellow and black tropical fish dart through the sea.
- An epic tornado attacking above aglowing city at night.
- Slow pan upward of blazing oak fire in an indoor fireplace.
- A cat wearing sunglasses and working as a lifeguard at pool.
- Sunset over the sea.
- A dog in astronaut suit and sunglasses floating in space.

<div><img src="docs/examples/latte_500steps.gif" width=80%></div>

</details>

### [DeepFake](#content)

#### 1.3.1 FaceSwap [![GitHub stars](https://img.shields.io/github/stars/deepfakes/faceswap?style=social&label=Star&maxAge=8640)](https://GitHub.com/deepfakes/faceswap/) 

``FaceSwap``, **a tool that utilizes deep learning to recognize and swap faces in pictures and videos.** FaceSwap supports various operating systems(``windows``, ``linux``, ``macos``) and offers powerful face swapping capabilities, utilizing a modern GPU with CUDA support for optimal performance. With FaceSwap, users can gather photos and videos, extract faces from them, train a model based on the extracted faces, and then seamlessly swap faces in your sources using the trained model. [url](https://faceswap.dev//), [code](https://github.com/deepfakes/faceswap)

#### 1.3.2 Roop [![GitHub stars](https://img.shields.io/github/stars/s0md3v/roop?style=social&label=Star&maxAge=8640)](https://GitHub.com/s0md3v/roop/) 

``Roop``, **a fantastic tool of taking a video and replace the face in it with a face of users' choices.** Users only need one image of the desired face. No dataset, no training. [code](https://github.com/s0md3v/roop), [repro](https://github.com/SJTU-TES/repro-roop)

<details>
<summary>Click to view examples we have implemented</summary>

<div style="display: flex; flex-direction: row;">
  <img src="docs/examples/emma.jpg" height="350px">
  <img src="docs/examples/roop_source.gif" height="350px">
  <img src="docs/examples/roop_output.gif" height="350px">
</div>

</details>

## [CO](#content)

### [Graph Matching](#content) 

#### 2.1.1 ⭐Pygmtools [![GitHub stars](https://img.shields.io/github/stars/Thinklab-SJTU/pygmtools?style=social&label=Star&maxAge=8640)](https://GitHub.com/Thinklab-SJTU/pygmtools/) 

``pygmtools``, **Python Graph Matching Tools, provides graph matching solvers in Python.** To make researchers' lives easier, pygmtools support various solvers (``linear``, ``quadratic``, ``multi-graph``, ``neural``), various backends (``numpy``, ``pytorch``, ``jittor``, ``paddle``, ``tensorflow``, ``mindspore``). Also, pygmtools is deep-learning-friendly, whose operations are designed to best preserve the gradient during computation and batched operations support for the best performance. [paper](https://jmlr.org/papers/volume25/23-0572/23-0572.pdf), [code](https://github.com/Thinklab-SJTU/pygmtools), [pretrained](https://huggingface.co/heatingma/pygmtools)

<details>
<summary>Click to view examples we have implemented</summary>
<img src="docs/examples/pygmtools.png" weight="700px">
</details>

### [Graph Edit Distance](#content)

#### 2.2.1 ⭐GENN-A* [![GitHub stars](https://img.shields.io/github/stars/Thinklab-SJTU/GENN-Astar?style=social&label=Star&maxAge=8640)](https://GitHub.com/Thinklab-SJTU/GENN-Astar/) 

``GENN-A*``, **Graph Edit Neural Network (GENN),** aims to accelerate the A* solver for graph edit distance problem based on Graph Neural Network. GENN-A* aided A* algorithm works by replacing the heuristic prediction module in A* by GNN. Since the accuracy of heuristic prediction is crucial for the performance of A*, this approach can significantly improve the efficiency of A*. [paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Wang_Combinatorial_Learning_of_Graph_Edit_Distance_via_Dynamic_Embedding_CVPR_2021_paper.pdf), [code](https://github.com/Thinklab-SJTU/GENN-Astar), [pretrained](https://drive.google.com/drive/folders/1mUpwHeW1RbMHaNxX_PZvD5HrWvyCQG8y)

<details>
<summary>Click to view examples we have implemented</summary>
<img src="docs/examples/genn-astar-1.png" weight="660px">
<img src="docs/examples/genn-astar-2.png" weight="440px">
</details>

### [Travelling Salesman Problem](#content)

#### 2.3.1 ⭐T2T [![GitHub stars](https://img.shields.io/github/stars/Thinklab-SJTU/T2TCO?style=social&label=Star&maxAge=8640)](https://GitHub.com/Thinklab-SJTU/T2TCO/) 

``T2T``, **Training to Testing.** T2TCO framework first leverages the generative modeling to estimate the high-quality solution distribution for each instance during training, and then conducts a gradient-based search within the solution space during testing. [paper](https://openreview.net/pdf?id=JtF0ugNMv2), [code](https://github.com/Thinklab-SJTU/T2TCO), [pretrained](https://drive.google.com/drive/folders/1IjaWtkqTAs7lwtFZ24lTRspE0h1N6sBH), [space](https://huggingface.co/spaces/SJTU-TES/TSP)

<details>
<summary>Click to view examples we have implemented</summary>
<div style="display: flex; flex-direction: row;">
  <img src="docs/examples/tsp_problem.png" height="250px">
  <img src="docs/examples/tsp_solution.png" height="250px">
</div>
</details>

### [Maximum Independent Set](#content)

#### 2.4.1 ⭐T2T [![GitHub stars](https://img.shields.io/github/stars/Thinklab-SJTU/T2TCO?style=social&label=Star&maxAge=8640)](https://GitHub.com/Thinklab-SJTU/T2TCO/)

``T2T``, **Training to Testing.** T2TCO framework first leverages the generative modeling to estimate the high-quality solution distribution for each instance during training, and then conducts a gradient-based search within the solution space during testing. [paper](https://openreview.net/pdf?id=JtF0ugNMv2), [code](https://github.com/Thinklab-SJTU/T2TCO), [pretrained](https://drive.google.com/drive/folders/1IjaWtkqTAs7lwtFZ24lTRspE0h1N6sBH), [space](https://huggingface.co/spaces/SJTU-TES/MIS)

<details>
<summary>Click to view examples we have implemented</summary>
<div style="display: flex; flex-direction: row;">
  <img src="docs/examples/mis_problem.png" height="250px">
  <img src="docs/examples/mis_solution.png" height="250px">
</div>
</details>

## [Website](#content)

### [Online Chatting](#content)

#### 3.1.1 ⭐GNetChat [![GitHub stars](https://img.shields.io/github/stars/heatingma/GNetChat?style=social&label=Star&maxAge=8640)](https://GitHub.com/heatingma/GNetChat/) 

``GNetChat``, **General Networking Chat Website designed by SJTUGN Group,** where students can easily form study groups, create posts, make friends, share essential resources, and collaborate on projects in real-time. [url](https://gnetchat.cn), [code](https://github.com/heatingma/GNetChat), [tutorial](https://github.com/heatingma/Chat-Website-Tutorial)

<details>
<summary>Click to view details</summary>
<img src="docs/examples/gnetchat1.png" width="80%" alt="" />
<img src="docs/examples/gnetchat2.png" width="80%" alt="" />
</details>

### [Web Scraping](#content)

#### 3.2.1 ⭐VidFetch [![GitHub stars](https://img.shields.io/github/stars/heatingma/VidFetch?style=social&label=Star&maxAge=8640)](https://GitHub.com/heatingma/VidFetch/) 

``VidFetch``, **an open-source dataset download tool to obtain copyright-free videos from various free video websites.** [code](https://github.com/heatingma/VidFetch)

<details>
<summary>Click to view details</summary>
<img src="docs/examples/VidFecth.png" width="80%" alt="" />
</details>

### [Cpp Online](#content)

#### 3.3.1 ⭐web-cpp [![GitHub stars](https://img.shields.io/github/stars/zhouzhouzhouxiao/web-cpp?style=social&label=Star&maxAge=8640)](https://GitHub.com/zhouzhouzhouxiao/web-cpp/) 

``web-cpp``, **an online platform that enables users to write and execute C++ code directly within their browsers.** [code](https://github.com/zhouzhouzhouxiao/web-cpp)

<details>
<summary>Click to view details</summary>
<img src="docs/examples/web-cpp.png" width="80%" alt="" />
</details>


## [Motion](#content)

### [Motion Retargeting](#content)

#### 4.1.1 Transmomo [![GitHub stars](https://img.shields.io/github/stars/yzhq97/transmomo.pytorch?style=social&label=Star&maxAge=8640)](https://GitHub.com/yzhq97/transmomo.pytorch) 

``Transmomo``, **Invariance-Driven Unsupervised Video Motion Retargeting** A lightweight video motion retargeting approach that is capable of transferring motion in spite of structural and view-angle disparities between the source and the target.[code](https://github.com/yzhq97/transmomo.pytorch)

<details>
<summary>Click to view details</summary>
<img src='https://yzhq97.github.io/assets/transmomo/dance.gif' width='480'/>
</p>
</details>

#### 4.1.2 EDN [![GitHub stars](https://img.shields.io/github/stars/carolineec/EverybodyDanceNow?style=social&label=Star&maxAge=8640)](https://github.com/carolineec/EverybodyDanceNow) 

``EverybodyDanceNow``, **A simple method for "do as I do" motion transfer:** Given a source video of a person dancing, we can transfer that performance to a novel (amateur) target after only a few minutes of the target subject performing standard moves.[code](https://github.com/carolineec/EverybodyDanceNow)

<details>
<summary>Click to view details</summary>
<img src="docs/examples/edn1.jpg" width="45%" alt="" />
<img src="docs/examples/edn2.jpg" width="45%" alt="" />
</details>

### [Pose Estimation](#content)

#### 4.2.1 Openpose [![GitHub stars](https://img.shields.io/github/stars/Hzzone/pytorch-openpose?style=social&label=Star&maxAge=8640)](https://github.com/Hzzone/pytorch-openpose) 

``Openpose``, **Real-time multi-person keypoint detection library for pose estimation** 2D real-time multi-person keypoint detection.We provide pytorch implementation of openpose including Body and Hand Pose Estimation.[code](https://github.com/Hzzone/pytorch-openpose)

<details>
<summary>Click to view details</summary>
<img src="docs/examples/openpose.jpg" width="60%" alt="" />
</details>

### [Video Matting](#content)
#### 4.3.1 RobustVideoMatting [![GitHub stars](https://img.shields.io/github/stars/PeterL1n/RobustVideoMatting?style=social&label=Star&maxAge=8640)](https://github.com/PeterL1n/RobustVideoMatting) 

``RVM``, **Robust High-Resolution Video Matting with Temporal Guidance** RVM is specifically designed for robust human video matting. Unlike existing neural models that process frames as independent images, RVM uses a recurrent neural network to process videos with temporal memory. [code](https://github.com/PeterL1n/RobustVideoMatting)

<details>
<summary>Click to view details</summary>
<img src="docs/examples/RVM.gif">
</details>

## [Security](#content)
### [DeepLearning Security](#content)

#### 5.1.1 ⭐DLSec [![GitHub stars](https://img.shields.io/github/stars/faiimea/DLSec?style=social&label=Star&maxAge=8640)](https://github.com/faiimea/DLSec) 

``DLSec``, **Deep Learning model security evaluation platform** Taking attack paradigms and defense means such as anti-sample, data poisoning, backdoor attacks as examples, We studies and implements mainstream offensive and defensive algorithms for deep learning models, and builds a comprehensive and effective evaluation system for deep learning models from the perspectives of white box model and black box model. [code](https://github.com/faiimea/DLSec)


#### 5.1.2 ⭐WDAD [![GitHub stars](https://img.shields.io/github/stars/faiimea/WDAD?style=social&label=Star&maxAge=8640)](https://github.com/faiimea/WDAD) 

``WDAD``, **Adversarial sample detection based on weak dark textures**  [code](https://github.com/faiimea/WDAD)

<details>
<summary>Click to view details</summary>
<img src="docs/examples/WDAD.png">
</details>

#### 5.1.3 ⭐UAP [![GitHub stars](https://img.shields.io/github/stars/faiimea/UAP?style=social&label=Star&maxAge=8640)](https://github.com/faiimea/UAP) 

``UAP``, **Fingerprinting Deep Neural Networks Globally via Universal Adversarial Perturbations"** a novel and practical mechanism which enables the service provider to verify whether a suspect model is stolen from the victim model via model extraction attacks. [code](https://github.com/faiimea/UAP)

### [IOT Security](#content)
#### 5.2.1 ⭐WAV2COM [![GitHub stars](https://img.shields.io/github/stars/faiimea/wav2com?style=social&label=Star&maxAge=8640)](https://github.com/faiimea/wav2com) 

``WAV2COM``, **Your Microphone Array Retains Your Identity: A Robust Voice Liveness Detection System for Smart Speakers**  [code](https://github.com/faiimea/wav2com)


## [Tool](#content)

### [ClearAgenda Scheduler](#content)
#### 6.1.1 ⭐ClearAgenda Scheduler [![GitHub stars](https://img.shields.io/github/stars/zhouzhouzhouxiao/ClearAgenda-Scheduler?style=social&label=Star&maxAge=8640)](https://github.com/zhouzhouzhouxiao/ClearAgenda-Scheduler) 

``ClearAgenda Scheduler``, **A simple daily management software based on tk, built for college students. It is very convenient for handling periodic events such as courses and visual schedules.**  [code](https://github.com/zhouzhouzhouxiao/ClearAgenda-Scheduler)
<details>
<summary>Click to view examples</summary>

<div style="display: flex; flex-direction: row;">
  <img src="docs/examples/ClearAgenda-Scheduler.png" height="350px">
</div>

</details>

