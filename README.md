# Medical AI Paper List
***
## Medical Class
### 1. REVIEW 综述类
[1] Shickel, Benjamin, et al. **Deep EHR: a survey of recent advances in deep learning techniques for electronic health record (EHR) analysis.** IEEE journal of biomedical and health informatics 22.5 (2017): 1589-1604.  
本地路径：F:\A_博士阶段\论文\Survey  
#### 主要内容：
本文调查了基于EHR数据的将深度学习应用于临床任务的最新研究，发现了应用于几个临床的深度学习技术和框架，包括信息提取，表示学习，结果预测，表型和去识别；并分析了当前研究的一些局限性，这些局限性涉及诸如模型可解释性，数据异质性和缺乏通用基准等主题。最后，我们总结了该领域的现状并确定了未来深入的EHR研究的途径。    

[2] Esteva, Andre, et al. **A guide to deep learning in healthcare.** Nature medicine 25.1 (2019): 24-29.  
本地路径：F:\A_博士阶段\论文\Survey   
#### 主要内容：
本文介绍了医疗保健领域的深度学习技术，重点讨论了计算机视觉（CV）、自然语言处理（NLP）、强化学习（RL）和通用方法方面的深度学习。我们将描述这些计算技术如何影响医学的几个关键领域，并探讨如何构建端到端系统。我们对CV的讨论主要集中在医学成像上，对NLP的描述主要在电子健康记录数据等领域。同样，在机器人辅助手术的背景下也讨论了RL，同时讨论了通用方法的深度学习在基因组学的应用。    

[3] Topol, Eric J. **High-performance medicine: the convergence of human and artificial intelligence.** Nature medicine 25.1 (2019): 44-56.  
#### 主要内容：
人工智能的使用，特别是深度学习，已经通过使用标记的大数据，以及显著增强的计算能力和云存储，实现了跨学科工作。在医学领域，深度学习开始在三个层面产生影响：对于临床医生，主要是快速、准确地解释图像；对卫生系统，改进工作流程和减少医疗错误的可能性；对病人，通过使他们能够处理自己的数据来保持健康。本文也讨论了当前的存在的局限性，包括偏见(bias)、隐私和安全性，缺乏透明度，以及这些应用的未来方向。   
[4] 

### 2. EHR Coding
[1] Xiancheng Xie, Yun Xiong, Philip S. Yu, Yangyong Zhu. **EHR Coding with Multi-scale Feature Attention and Structured Knowledge Graph Propagation**. CIKM 2019: 649-658 [[PDF]](http://delivery.acm.org/10.1145/3360000/3357897/p649-xie.pdf?ip=211.87.239.55&id=3357897&acc=OPEN&key=BF85BBA5741FDC6E%2EBA9BBD89F2E1EC6A%2E4D4702B0C3E38B35%2E6D218144511F3437&__acm__=1575427234_a9f472fe217137daaa87426759aa5dc1)  
本地路径：F:\A_博士阶段\论文\CIKM2019\医疗健康相关

### 3. 消化内镜
[1] Wu, Lianlian, et al. **Randomised controlled trial of WISENSE, a real-time quality improving system for monitoring blind spots during esophagogastroduodenoscopy.** Gut (2019): gutjnl-2018. [[PDF]](https://gut.bmj.com/content/gutjnl/68/12/2161.full.pdf)  
本地路径：F:\A_博士阶段\消化内镜资料\论文  
#### 主要内容：
食管胃十二指肠镜（egd）是诊断上消化道病变的关键。然而，内镜医师在egd表现上存在显著差异，影响胃癌和前体病变的发现率。本研究的目的是建立一个实时的质量改进系统WISENSE，在EGD过程中监测盲点，对过程进行计时，并自动生成光图像，从而提高日常内镜检查的质量。  
利用深度卷积神经网络和深度强化学习方法开发了wisense系统。患者包括武汉大学人民医院因体检、症状、监护等原因转诊的病人。入选的患者被随机分为在WISENSE的帮助下或不帮助下接受EGD的组。主要目的是确定WISENSE辅助组与对照组之间的盲点率是否存在差异。  
Wisense在真实的EGD视频中监控盲点的准确率为90.40%。共有324名患者被招募并随机化。wisense组153例，对照组150例。与对照组相比，wisense组的盲点率较低（5.86%对22.46%，p<0.001），平均差异为-15.39%（95%ci-19.23至-11.54）。无明显不良反应。WISENSE显著降低了EGD手术的盲点率，可用于提高日常内镜检查的质量。

[2] Luo, Huiyan, et al. **Real-time artificial intelligence for detection of upper gastrointestinal cancer by endoscopy: a multicentre, case-control, diagnostic study**. The Lancet Oncology 20.12 (2019): 1645-1654. [[PDF]](https://gut.bmj.com/content/gutjnl/68/12/2161.full.pdf)  
本地路径：F:\A_博士阶段\消化内镜资料\论文  
#### 主要内容：
该系统使用深度卷积神经网络（主要采用了DeepLab’s V3：图像语义分割，机器自动从图像中分割出对象区域，并识别其中的内容）针对内镜图像部位多、疾病种类多、癌变表现多样化情况下实现高准确性识别与标记，能够在高度复杂人机协同的临床实操环境中实现稳定的预测；同时该系统的速度非常快，一台配置单GPU卡的普通服务器即可达到每秒118张图像的处理能力，处理延时低于10ms。  
该智能辅助系统具有实时活检部位精确提示、内镜检查智能质控和自动采图等功能，在医生进行内镜检查的同时自动捕获图像并进行云端AI分析，实时提示精确的可疑病灶区域，指导内镜医生选择活检部位；在检查过程中，该系统能对检查时间和检查部位进行质控，减少遗漏关键信息，提高检查质量；在临床操作中，该系统还能够依据指南要求自动采图存储，减少医生“一心两用”、“手脚并用”带来遗漏关键信息的可能性。  
构建了基于云技术的多中心上消化道癌内镜AI诊断平台，该云诊断平台可以自动捕获内镜检查图像上传至云端进行AI分析，实时向操作者反馈提示可疑病灶区域，指导操作者更有针对性的选择活检部位，提高活检阳性率。  

[3] Hirasawa, Toshiaki, et al. **Application of artificial intelligence using a convolutional neural network for detecting gastric cancer in endoscopic images.** Gastric Cancer 21.4 (2018): 653-660.  
#### 主要内容：
利用卷积神经网络（cnns）进行深度学习的人工智能图像识别技术得到了极大的改进，并越来越多地应用于医学诊断成像领域。我们开发了一种能在内镜图像中自动检测胃癌的cnn。  
构建基于cnn的胃癌诊断系统，该CNN结构基于Single Shot MultiBox Detector（SSD），利用13584张胃癌内镜图像进行训练。为了评估诊断的准确性，我们将收集自69例患者77个胃癌病灶的2296个胃图像的独立测试集应用于构建的cnn。  
cnn需要47秒分析2296张测试图像。cnn对77例胃癌病变中71例诊断正确，总敏感度为92.2%，对161例非癌病变检出胃癌，阳性预测值为30.6%。71个直径大于等于6mm的病灶中，70个（98.6%）及所有浸润性癌均被正确检出。所有漏诊病灶均为浅表凹陷型和分化型粘膜内癌，即使是经验丰富的内镜医师也难以与胃炎区分。假阳性病变中近一半为胃炎，颜色改变或粘膜表面不规则。  
构建的cnn胃癌检测系统能够在很短的时间内处理大量存储的内镜图像，具有临床相关的诊断能力。它可以很好地应用于日常临床实践，以减轻内镜医师的负担。



***
## CS Class
### 1. Knowledge Graph
[1] Xiancheng Xie, Yun Xiong, Philip S. Yu, Yangyong Zhu. **EHR Coding with Multi-scale Feature Attention and Structured Knowledge Graph Propagation**. CIKM 2019: 649-658 [[PDF]](http://delivery.acm.org/10.1145/3360000/3357897/p649-xie.pdf?ip=211.87.239.55&id=3357897&acc=OPEN&key=BF85BBA5741FDC6E%2EBA9BBD89F2E1EC6A%2E4D4702B0C3E38B35%2E6D218144511F3437&__acm__=1575427234_a9f472fe217137daaa87426759aa5dc1)  
本地路径：F:\A_博士阶段\论文\CIKM2019\医疗健康相关


***
## Research Group
### Knowledge Graph
[1] fudan university+ university of illinois at chicago  
Xiancheng Xie, Yun Xiong, Philip S Yu(uic), Yangyong Zhu
