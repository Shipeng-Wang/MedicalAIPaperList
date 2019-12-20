# Medical Survey
***
## Medical Classification
### 1. REVIEW and SURVEY 综述类
[1] Shickel, Benjamin, et al. **Deep EHR: a survey of recent advances in deep learning techniques for electronic health record (EHR) analysis.** IEEE journal of biomedical and health informatics 22.5 (2017): 1589-1604.  
本地路径：F:\A_博士阶段\论文\Survey  
**主要内容**：  
本文调查了基于EHR数据的将深度学习应用于临床任务的最新研究，发现了应用于几个临床的深度学习技术和框架，包括信息提取，表示学习，结果预测，表型和去识别；并分析了当前研究的一些局限性，这些局限性涉及诸如模型可解释性，数据异质性和缺乏通用基准等主题。最后，我们总结了该领域的现状并确定了未来深入的EHR研究的途径。    

[2] Esteva, Andre, et al. **A guide to deep learning in healthcare.** Nature medicine 25.1 (2019): 24-29.  
本地路径：F:\A_博士阶段\论文\Survey   
**主要内容**：  
本文介绍了医疗保健领域的深度学习技术，重点讨论了计算机视觉（CV）、自然语言处理（NLP）、强化学习（RL）和通用方法方面的深度学习。我们将描述这些计算技术如何影响医学的几个关键领域，并探讨如何构建端到端系统。我们对CV的讨论主要集中在医学成像上，对NLP的描述主要在电子健康记录数据等领域。同样，在机器人辅助手术的背景下也讨论了RL，同时讨论了通用方法的深度学习在基因组学的应用。    

[3] Topol, Eric J. **High-performance medicine: the convergence of human and artificial intelligence.** Nature medicine 25.1 (2019): 44-56.  
本地路径：F:\A_博士阶段\论文\Survey  
**主要内容**：  
人工智能的使用，特别是深度学习，已经通过使用标记的大数据，以及显著增强的计算能力和云存储，实现了跨学科工作。在医学领域，深度学习开始在三个层面产生影响：对于临床医生，主要是快速、准确地解释图像；对卫生系统，改进工作流程和减少医疗错误的可能性；对病人，通过使他们能够处理自己的数据来保持健康。本文也讨论了当前的存在的局限性，包括偏见(bias)、隐私和安全性，缺乏透明度，以及这些应用的未来方向。   

### 2. Embedding
#### 2.1 Medical Concept Embedding
[1] Peng, Xueping, et al. **Temporal self-attention network for medical concept embedding.** arXiv preprint arXiv:1909.06886 (2019).[[PDF]](https://arxiv.org/pdf/1909.06886.pdf)  
**主要内容**：   
在纵向电子健康记录（EHRs）中，患者的事件记录分布在很长一段时间内，事件之间的时间关系反映了足够的领域知识，有助于进行住院死亡率等预测任务。医学概念嵌入作为一种特征提取方法，将一组具有特定时间戳的医学概念转化为一个向量，并将其输入到一个有监督的学习算法中。嵌入的质量对医学数据的学习性能有显著的影响。本文提出了一种基于自我注意机制的医学概念嵌入方法来表示每一个医学概念。我们提出了一种新的注意机制来捕捉医学概念之间的上下文信息和时间关系。在此基础上，提出了一种轻量化的神经网络“时间自关注网络（TeSAN）”，用于学习基于所提出的注意机制的医学概念嵌入。为了验证我们提出的方法的有效性，我们对两个公共EHRs数据集进行了聚类和预测任务，将TeSAN与五种最新的嵌入方法进行了比较。实验结果表明，本文提出的TeSAN模型优于所有比较方法。据我们所知，这项工作是第一次利用医学事件之间的时间自我关注关系。

#### 2.2 EHR Coding
[1] Xiancheng Xie, Yun Xiong, Philip S. Yu, Yangyong Zhu. **EHR Coding with Multi-scale Feature Attention and Structured Knowledge Graph Propagation**. CIKM 2019: 649-658 [[PDF]](http://delivery.acm.org/10.1145/3360000/3357897/p649-xie.pdf?ip=211.87.239.55&id=3357897&acc=OPEN&key=BF85BBA5741FDC6E%2EBA9BBD89F2E1EC6A%2E4D4702B0C3E38B35%2E6D218144511F3437&__acm__=1575427234_a9f472fe217137daaa87426759aa5dc1)  
本地路径：F:\A_博士阶段\论文\CIKM2019\医疗健康相关   
**主要内容**： 
背景：将代表诊断或程序的标准医疗代码（如ICD-9-CM）分配给电子健康记录（EHR）是医学领域的一项重要任务。  
问题：然而，由于临床笔记是由多个长而异质的文本叙述（如出院诊断、病理报告、外科手术笔记）组成，因此自动编码很困难。此外，编码标签空间大，标签分布极不平衡。目前的方法主要是将EHR编码作为一个多标签文本分类任务，采用固定窗口大小的浅卷积神经网络，无法学习可变的n-gram特征和代码之间的本体结构。  
本文工作：（1）在本文中，我们利用一个紧密连接的卷积神经网络，它能够产生可变的n-gram特征，用于临床笔记特征的学习。（2）我们还结合了多尺度特征注意来自适应地选择多尺度特征，因为每个单词的临床笔记中信息量最大的n-图可以根据邻域而变化长度。（3）我们利用图卷积神经网络来捕捉医疗编码之间的层次关系和每个编码的语义。最后，我们在公共数据集上验证了我们的方法，评估结果表明我们的方法可以显著优于其他最新的模型。  

[2] Mullenbach, James, et al. **Explainable prediction of medical codes from clinical text.** arXiv preprint arXiv:1802.05695 (2018).[[PDF]](https://arxiv.org/pdf/1802.05695.pdf)  
**主要内容**：  


### 3. Prediction  
[1] Gao, Jingyue, et al. **Camp: Co-attention memory networks for diagnosis prediction in healthcare.** ICDM, 2019. [[PDF]](https://jygao97.github.io/papers/CAMP_ICDM19_long.pdf)  
本地路径：F:\A_博士阶段\论文\CIKM2019\医疗健康相关  
**主要内容**：  
诊断预测是个性化医疗的核心研究课题，其目的是从历史电子病历中预测患者未来的健康信息。虽然已有一些基于RNN的方法被提出用于序列EHR数据的建模，但这些方法存在三个主要问题。首先，他们无法捕捉到患者健康状况的细粒度发展模式。第二，他们没有考虑重要背景（例如，患者人口统计）和历史诊断之间的相互影响。第三，RNN中隐藏的状态向量难以解释，导致信任问题。  
为了应对这些挑战，我们提出了一个被称为诊断预测共同注意记忆网络（CAMP）的模型，该模型将历史记录、细粒度患者状况和人口统计学与基于共同注意的三方交互架构紧密地结合在一起。我们的模型使用一个存储网络来扩充RNN，以丰富表示能力。内存网络通过显式地将疾病分类合并到一个内存槽数组中，实现了对细粒度患者状况的分析。我们设计了内存槽以确保可解释性，并实例化了内存网络的读/写操作，从而通过共同关注，使内存与患者统计数据有效地协同工作。实验和对真实数据集的案例研究表明，CAMP在预测精度方面始终优于最新的方法，并且具有很高的可解释性。  

[2] Tan, Qingxiong, et al. **UA-CRNN: Uncertainty-Aware Convolutional Recurrent Neural Network for Mortality Risk Prediction.** Proceedings of the 28th ACM International Conference on Information and Knowledge Management. ACM, 2019.[[PDF]](http://delivery.acm.org/10.1145/3360000/3357884/p109-tan.pdf?ip=211.87.239.55&id=3357884&acc=ACTIVE%20SERVICE&key=BF85BBA5741FDC6E%2EBA9BBD89F2E1EC6A%2E4D4702B0C3E38B35%2E4D4702B0C3E38B35&__acm__=1576813028_16f942d129d2e60f5bd96ca1174e62a4)  
**主要内容**：  
准确预测死亡风险对于评估早期治疗、发现高危患者和改善医疗保健效果具有重要意义。由于连续记录中的时间间隔不同，从不规则的临床时间序列数据预测死亡率风险具有挑战性。现有的方法通常通过从原始的不规则数据生成规则的时间序列数据来解决这个问题，而不考虑由变化的时间间隔引起的生成数据中的不确定性。本文提出了一种新的不确定性感知卷积递归神经网络（UA-CRNN），它将不确定性信息融入到生成的数据中，以提高死亡率风险预测性能。为了处理具有不同频率子序列的复杂临床时间序列数据，我们建议将不确定性信息纳入子序列层次，而不是整个时间序列数据。具体地说，我们设计了一种新的分层不确定性感知分解层（UADL），将时间序列自适应地分解成不同的子序列，并根据其可靠性赋予相应的权重。在两个实际临床数据集上的实验结果表明，所提出的UA-CRNN方法在短期和长期死亡率风险预测方面均显著优于最新方法。

[3] Yin, Changchang, et al. **Domain Knowledge Guided Deep Learning with Electronic Health Records.** ICDM, 2019.[[PDF]](https://pdfs.semanticscholar.org/7007/fb3b4c316b2146057f2ba12f3cf0ba5dcbd0.pdf)  
**主要内容**：  
由于其在电子健康记录（EHRs）临床风险预测方面的良好表现，深度学习方法引起了医疗研究人员的极大兴趣。然而，有四个挑战：（一）数据不足。许多方法需要大量的训练数据才能达到满意的效果。（二）可解释性。许多方法的结果很难向临床医生解释（例如，为什么模型会做出特定的预测，哪些事件会导致临床结果）。（三）领域知识整合。没有现有的方法动态地利用复杂的医学知识（例如，因果关系，并且是由临床事件之间引起的）。（四）时间间隔信息。大多数现有方法只考虑EHR访问的相对顺序，而忽略相邻访问之间的不规则时间间隔。在本研究中，我们提出一个新的模式，即领域知识引导回归神经网路（DG-RNN），直接将医学知识图中的领域知识引入RNN架构，并考虑不规则的时间间隔。心衰风险预测任务的实验结果表明，我们的模型不仅优于目前最先进的基于深度学习的风险预测模型，而且还将个体医疗事件与心衰发作相关联，从而为可解释的准确临床风险预测铺平了道路。  

### 4. Classification
[1] Li, Xiaoyu, et al. **Domain Knowledge Guided Deep Atrial Fibrillation Classification and Its Visual Interpretation.** Proceedings of the 28th ACM International Conference on Information and Knowledge Management. ACM, 2019.[[PDF]](http://delivery.acm.org/10.1145/3360000/3357998/p129-li.pdf?ip=211.87.239.55&id=3357998&acc=ACTIVE%20SERVICE&key=BF85BBA5741FDC6E%2EBA9BBD89F2E1EC6A%2E4D4702B0C3E38B35%2E4D4702B0C3E38B35&__acm__=1576813170_a9dbbee15ab45ae129d8641a68bb7e70)  
**主要内容**：  
手工制作的特征已经被证明在解决心电图分类问题上是有用的。这些特征依赖于领域知识，具有临床意义。然而，特性的构造在实践中需要冗长的微调。近年来，人们提出了一套端到端的深层神经网络模型，在心电分类中取得了很好的效果。这些模型虽然有效，但往往学习到与人的概念不匹配的模式，因此很难用解释方法得到令人信服的解释。考虑到心脏病专家难以接受深度学习无法解释的结果，这种局限性显著地缩小了深度模型的适用范围。为了缓解这种局限性，我们从两个世界中汲取精华，提出了一种基于领域知识的深度神经网络。具体地说，我们利用一个深度残差网络作为分类框架，在这个框架中，我们采用关键特征（P波和R峰位置）重建任务，将领域知识融入到学习过程中。重建任务使得模型更加关注心电信号中的关键特征点。此外，利用遮挡方法进行视觉解译，设计了心跳级和特征点级的可视化。实验结果表明，与无P波和R峰任务的模型相比，本文提出的ECG分类方法具有更好的分类性能，所得到的模式更易于解释。

### 5. Knowledge Discovering
[1] Deng, Yang, et al. **MedTruth: A Semi-supervised Approach to Discovering Knowledge Condition Information from Multi-Source Medical Data.** Proceedings of the 28th ACM International Conference on Information and Knowledge Management. ACM, 2019.[[PDF]](http://delivery.acm.org/10.1145/3360000/3357934/p719-deng.pdf?ip=211.87.239.55&id=3357934&acc=ACTIVE%20SERVICE&key=BF85BBA5741FDC6E%2EBA9BBD89F2E1EC6A%2E4D4702B0C3E38B35%2E4D4702B0C3E38B35&__acm__=1576812887_19dd1480132a90b9995fe5a8c4fa37ff)  
**主要内容**：  
知识图（KG）包含实体和实体之间的关系。由于它的表示能力，KG已经成功地应用于支持许多医疗/保健任务。然而，在医学领域，知识是在一定条件下存在的。例如，当患者是婴儿而不是其他年龄的人时，症状流鼻涕高度指示疾病百日咳的存在。医学知识的这种条件对于各种医学应用中的决策是至关重要的，在现有的医疗KGS中缺失。本文旨在从文本中发现医学知识条件，丰富知识库。  
电子病历是临床数据的系统化收集，包含病人的详细信息，是发现医学知识状况的良好资源。不幸的是，由于正规化等原因，可用的电子病历数量有限。同时，大量的医学问答（QA）数据可供参考，对课题的研究有很大的帮助。然而，医疗质量保证数据的质量参差不齐，这可能会降低发现医疗知识状况的质量。针对这些挑战，我们提出了一种新的真理发现方法medthuth，用于医学知识条件的发现，它将先验的信源质量信息纳入信源可靠性估计过程中，并利用知识三重信息进行可信信息计算，在真实的医学数据集上进行了一系列实验，证明该方法能同时利用EMR和QA数据发现有意义和准确的医学知识条件。此外，在合成数据集上验证了该方法在不同场景下的有效性。


## 消化疾病
[1] Wu, Lianlian, et al. **Randomised controlled trial of WISENSE, a real-time quality improving system for monitoring blind spots during esophagogastroduodenoscopy.** Gut (2019): gutjnl-2018. [[PDF]](https://gut.bmj.com/content/gutjnl/68/12/2161.full.pdf)  
本地路径：F:\A_博士阶段\消化内镜资料\论文    
**主要内容**：  
食管胃十二指肠镜（egd）是诊断上消化道病变的关键。然而，内镜医师在egd表现上存在显著差异，影响胃癌和前体病变的发现率。本研究的目的是建立一个实时的质量改进系统WISENSE，在EGD过程中监测盲点，对过程进行计时，并自动生成光图像，从而提高日常内镜检查的质量。  
利用深度卷积神经网络和深度强化学习方法开发了wisense系统。患者包括武汉大学人民医院因体检、症状、监护等原因转诊的病人。入选的患者被随机分为在WISENSE的帮助下或不帮助下接受EGD的组。主要目的是确定WISENSE辅助组与对照组之间的盲点率是否存在差异。  
Wisense在真实的EGD视频中监控盲点的准确率为90.40%。共有324名患者被招募并随机化。wisense组153例，对照组150例。与对照组相比，wisense组的盲点率较低（5.86%对22.46%，p<0.001），平均差异为-15.39%（95%ci-19.23至-11.54）。无明显不良反应。WISENSE显著降低了EGD手术的盲点率，可用于提高日常内镜检查的质量。  

[2] Luo, Huiyan, et al. **Real-time artificial intelligence for detection of upper gastrointestinal cancer by endoscopy: a multicentre, case-control, diagnostic study**. The Lancet Oncology 20.12 (2019): 1645-1654. [[PDF]](https://gut.bmj.com/content/gutjnl/68/12/2161.full.pdf)  
本地路径：F:\A_博士阶段\消化内镜资料\论文  
**主要内容**：  
该系统使用深度卷积神经网络（主要采用了DeepLab’s V3：图像语义分割，机器自动从图像中分割出对象区域，并识别其中的内容）针对内镜图像部位多、疾病种类多、癌变表现多样化情况下实现高准确性识别与标记，能够在高度复杂人机协同的临床实操环境中实现稳定的预测；同时该系统的速度非常快，一台配置单GPU卡的普通服务器即可达到每秒118张图像的处理能力，处理延时低于10ms。  
该智能辅助系统具有实时活检部位精确提示、内镜检查智能质控和自动采图等功能，在医生进行内镜检查的同时自动捕获图像并进行云端AI分析，实时提示精确的可疑病灶区域，指导内镜医生选择活检部位；在检查过程中，该系统能对检查时间和检查部位进行质控，减少遗漏关键信息，提高检查质量；在临床操作中，该系统还能够依据指南要求自动采图存储，减少医生“一心两用”、“手脚并用”带来遗漏关键信息的可能性。  
构建了基于云技术的多中心上消化道癌内镜AI诊断平台，该云诊断平台可以自动捕获内镜检查图像上传至云端进行AI分析，实时向操作者反馈提示可疑病灶区域，指导操作者更有针对性的选择活检部位，提高活检阳性率。  

[3] Hirasawa, Toshiaki, et al. **Application of artificial intelligence using a convolutional neural network for detecting gastric cancer in endoscopic images.** Gastric Cancer 21.4 (2018): 653-660. 
本地路径：F:\A_博士阶段\消化内镜资料\论文  
**主要内容**：  
利用卷积神经网络（cnns）进行深度学习的人工智能图像识别技术得到了极大的改进，并越来越多地应用于医学诊断成像领域。我们开发了一种能在内镜图像中自动检测胃癌的cnn。  
构建基于cnn的胃癌诊断系统，该CNN结构基于Single Shot MultiBox Detector（SSD），利用13584张胃癌内镜图像进行训练。为了评估诊断的准确性，我们将收集自69例患者77个胃癌病灶的2296个胃图像的独立测试集应用于构建的cnn。  
cnn需要47秒分析2296张测试图像。cnn对77例胃癌病变中71例诊断正确，总敏感度为92.2%，对161例非癌病变检出胃癌，阳性预测值为30.6%。71个直径大于等于6mm的病灶中，70个（98.6%）及所有浸润性癌均被正确检出。所有漏诊病灶均为浅表凹陷型和分化型粘膜内癌，即使是经验丰富的内镜医师也难以与胃炎区分。假阳性病变中近一半为胃炎，颜色改变或粘膜表面不规则。  
构建的cnn胃癌检测系统能够在很短的时间内处理大量存储的内镜图像，具有临床相关的诊断能力。它可以很好地应用于日常临床实践，以减轻内镜医师的负担。



***
## CS Classification
### 1. Knowledge Graph
[1] Xiancheng Xie, Yun Xiong, Philip S. Yu, Yangyong Zhu. **EHR Coding with Multi-scale Feature Attention and Structured Knowledge Graph Propagation**. CIKM 2019: 649-658 [[PDF]](http://delivery.acm.org/10.1145/3360000/3357897/p649-xie.pdf?ip=211.87.239.55&id=3357897&acc=OPEN&key=BF85BBA5741FDC6E%2EBA9BBD89F2E1EC6A%2E4D4702B0C3E38B35%2E6D218144511F3437&__acm__=1575427234_a9f472fe217137daaa87426759aa5dc1)  
本地路径：F:\A_博士阶段\论文\CIKM2019\医疗健康相关  

### 1. Attention Mechanism 
[1] Xiancheng Xie, Yun Xiong, Philip S. Yu, Yangyong Zhu. **EHR Coding with Multi-scale Feature Attention and Structured Knowledge Graph Propagation**. CIKM 2019: 649-658 [[PDF]](http://delivery.acm.org/10.1145/3360000/3357897/p649-xie.pdf?ip=211.87.239.55&id=3357897&acc=OPEN&key=BF85BBA5741FDC6E%2EBA9BBD89F2E1EC6A%2E4D4702B0C3E38B35%2E6D218144511F3437&__acm__=1575427234_a9f472fe217137daaa87426759aa5dc1)  
本地路径：F:\A_博士阶段\论文\CIKM2019\医疗健康相关  

[2] Gao, Jingyue, et al. **Camp: Co-attention memory networks for diagnosis prediction in healthcare.** ICDM, 2019. [[PDF]](https://jygao97.github.io/papers/CAMP_ICDM19_long.pdf)  
本地路径：F:\A_博士阶段\论文\CIKM2019\医疗健康相关


***
## Research Group
### Knowledge Graph
[1] fudan university+ university of illinois at chicago  
Xiancheng Xie, Yun Xiong, Philip S Yu(uic), Yangyong Zhu
