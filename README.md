# OpenSetGAD

Social Event Detection (SED) has gained considerable research interest in recent years [1, 10], as evidenced by the plethora of methodologies developed to address this challenge. Existing techniques can be broadly categorized into two types: non-graph-based SED and graph-based SED.

Non-graph-based SED methods were the early pioneers in the field, exploring statistical features such as term frequencies and word co-occurrences [6, 21], and leveraging topic models to discover latent themes from diverse attributes of unstructured data [32-34, 37]. More recently, studies have evolved to harness the power of deep neural networks, specifically recurrent neural networks (RNNs) and convolutional neural networks (CNNs), to efficiently extract both semantic and syntactic information within textual data [2, 13, 22, 31].

Graph-based SED methods, on the other hand, have introduced structural information and rich semantics to event detection. Early research utilized community detection techniques to identify groups and activities within social media data. In recent years, there has been a significant surge in the development of Graph Neural Networks (GNNs)-based SED models to address the limitations of traditional methods. These methodologies typically model social media streams as graphs, enabling GNNs to effectively capture the complex semantic and structural relationships within the data, thereby enhancing the accuracy and robustness of social event detection [5, 12, 18].

Both approaches have their merits and limitations, with non-graph-based methods providing simpler and faster analyses, and graph-based approaches offering more nuanced insights into the social structure and context surrounding the events in question. As the field of SED continues to mature, the integration of these methodologies, coupled with advancements in machine learning and graph theory, is expected to lead to even more sophisticated and effective solutions for social event detection in the digital age.

However, these approaches have limitations. Incremental clustering methods can be easily adapted to open-set social event detection tasks, but they fail to fully leverage the rich semantics and structural information in social streams. 

Recently proposed method QSGNN [26] provides an effective
solution for SED in an open-set setting (i.e. open-set SED), which
aims to identify newly emerging events in unlabeled incremental
social streams with the support of knowledge learned from existing
known events.

In recent years, Graph Neural Networks (GNNs) have emerged as a focal point of research interest, largely attributed to their proficiency in adeptly capturing intricate relationships embedded within graph-formulated data. The landscape of GNN methodologies has seen a proliferation of diverse models, notably encompassing Graph Convolutional Networks (GCNs) [9], Graph Attention Networks (GATs) [44], and GraphSAGE [17]. These architectures have found broad application across a spectrum of tasks, spanning from node classification, graph classification, to link prediction, illustrating their versatility and effectiveness in handling graph-based problems.
[9]Learning Convolutional Neural Networks for Graphs
[44]Graph attention networks. In International Conference on Learning Representations.
[17]Inductive Representation Learning on Large Graphs


The burgeoning field of graph representation learning, encompassing early graph embedding techniques [12, 33, 41] and the recent proliferation of Graph Neural Networks (GNNs) [13, 20, 43, 50], has unlocked significant potential for a wide array of downstream tasks, both at the node and graph levels. It's noteworthy that crafting representations at the graph level necessitates an additional step known as the ReadOut operation. This operation consolidates a graph's global information by amassing the representations of individual nodes using either flat [8, 11, 50, 56] or hierarchical [10, 21, 31, 51] pooling strategies, enabling a holistic understanding of graph structure and function.
[12]node2vec: Scalable feature learning for networks.
[33]DeepWalk: Online learning of social representations.
[41]LINE: Large-scale information network embedding
[13]Inductive representation learning on large graphs.
[20]Semi-supervised classification with graph convolutional networks
[43]Graph attention networks. In International Conference on Learning Representations.
[50]How powerful are graph neural networks?
[10]Graph u-nets. In International Conference on Machine Learning.
[21]Self-attention graph pooling. In International Conference on Machine Learning
[31]Graph convolutional networks with eigenpooling
[51]Hierarchical graph representation learning with differentiable pooling.


The adoption of the pre-training and fine-tuning framework in GNNs facilitates the training of models on large-scale graph datasets, with or without labels, before fine-tuning for specific downstream tasks. This two-step process significantly improves model initialization, providing broader optima and enhanced generalization capabilities compared to training models from scratch, thus addressing the limitations posed by scarce labeled data and enhancing model performance.

There are three main categories of pre-training strategies for GNNs: node-level, edge-level, and graph-level. Node-level Pre-training emphasizes local structure representations, using contrastive learning and predictive models to maximize Mutual Information between original and perturbed views[6, 123, 37, 62, 33,]. However, it struggles to capture higher-order information, focusing mainly on local topology. Edge-level Pre-training Aimed at enhancing link prediction tasks, these strategies utilize contrastive learning to predict edge presence and adjacency matrix reconstruction[82, 41, 58, 23, 39]. While strong in structural understanding, they overlook node properties, limiting effectiveness in graph-level tasks. Graph-level Pre-training focus on graph-level representations for subgraph tasks, this approach includes graph reconstruction and contrastive methods to encode global information. Challenges arise when transferring this knowledge to tasks with significant structural differences, potentially leading to negative transfer[88, 76, 24, 69].
6, Wiener graph deconvolutional network improves graph self-supervised learning
123, Graph Contrastive Learning with Adaptive Augmentation
37, Multi-Scale Contrastive Siamese Networks for Self-Supervised Graph Representation Learning
62, Graph Representation Learning via Graphical Mutual Information Maximization
82,  
58, 
23, 
39
88, 
76, 
24, 
69

There are three primary approaches to graph prompt tuning: Meta-Learning Based Approaches: These methods utilize meta-learning techniques, such as Model-Agnostic Meta-Learning (MAML), to learn a robust starting point for prompt parameters. This enables the learned graph prompt to be adaptable across a wide range of downstream tasks. Task-Specific Tuning: This approach focuses on tuning the graph prompt specifically for individual tasks. For instance, the Graph Prompt Tuning (GPF) method [11] targets graph classification tasks. It achieves this by optimizing the prompt token and task head to maximize the likelihood of accurately predicting graph labels. Tuning in Line with Pretext: Aligning prompt tuning with the pre-training task by adopting similar loss functions or objectives, such as GraphPrompt [52], GPPT [78] using the same loss function as the downstream task or employing node-level contrastive loss.

Recently, the task of novel class discovery has been proposed, which aims to identify novel classes in unlabeled data[18], similar to our open set event detection task. Unlike traditional unsupervised learning methods, this task leverages known data to facilitate the discovery of new classes. Existing methods[17–20, 51, 52] typically employ a two-stage approach: (1) initializing a model using labeled data, and (2) fine-tuning the model through unsupervised clustering or pairwise determination on unlabeled data. For instance, authors in [19] proposed a constrained clustering network to recognize open set images, which first measures pairwise similarities between images using a classification model trained on labeled data, and then applies a clustering model to unlabeled data using these pairwise predictions. Similarly, authors in [17] use rank statistics to estimate pairwise similarity directly. Although these methods, which use pseudo-labels to adapt the model to unlabeled data, have achieved promising results, they overlook the noise present in the obtained pseudo-labels, making the training process unreliable. Furthermore, they fail to provide a strategy for selecting samples. Given the large amount of unlabeled data, selecting a small number of samples to improve performance has become a critical issue.

[17]Automatically discovering and learning new visual categories with ranking statistics.
[18]Learning to discover novel visual categories via deep transfer clustering
[19]Multi-class classification without multi-class labels. In International Conference on Learning Representations.
[20]Joint representation learning and novel category discovery on single-and multi-modal data.
[51]Neighborhood Contrastive Learning for Novel Class Discovery 
[52]Openmix: Reviving known knowledge for discovering novel visual categories in an open world.


REFERENCES
[1] A. M. Bran and P. Schwaller, “Transformers and Large
Language Models for Chemistry and Drug Discovery,”
arXiv preprint, 2023.
[2] T. Brown, B. Mann, N. Ryder, M. Subbiah, J. D. Kaplan,
P. Dhariwal et al., “Language Models Are Few-Shot
Learners,” in NeurIPS, vol. 33, 2020, pp. 1877–1901.
[3] Y. Cao, J. Xu, C. Yang, J. Wang, Y. Zhang, C. Wang,
L. Chen, and Y. Yang, “When to pre-train graph
neural networks? An answer from data generation
perspective!” in KDD, 2023.
[4] M. Chen, Z. Liu, C. Liu, J. Li, Q. Mao, and J. Sun,
“ULTRA-DP: Unifying Graph Pre-training with Multitask
Graph Dual Prompt,” arXiv preprint, 2023.
[5] Z. Chen, H. Mao, H. Li, W. Jin, H. Wen, X. Wei,
S.Wang, D. Yin,W. Fan, H. Liu, and J. Tang, “Exploring
the Potential of Large Language Models (LLMs) in
Learning on Graphs,” arXiv preprint, 2023.
[6] J. Cheng, M. Li, J. Li, and F. Tsung, “Wiener graph deconvolutional
network improves graph self-supervised
learning,” in AAAI, 2023, pp. 7131–7139.
[7] E. Dai and S. Wang, “Towards Self-Explainable Graph
Neural Network,” in CIKM, 2021, pp. 302–311.
[8] M. Defferrard, X. Bresson, and P. Vandergheynst,
“Convolutional Neural Networks on Graphs with Fast
Localized Spectral Filtering,” in NeurIPS, vol. 29, 2016.
[9] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova,
“BERT: Pre-training of Deep Bidirectional Transformers
for Language Understanding,” arXiv preprint, 2019.
[10] C. Edwards, A. Naik, T. Khot, M. Burke, H. Ji, and
T. Hope, “SynerGPT: In-Context Learning for Personalized
Drug Synergy Prediction and Drug Design,” arXiv
preprint, 2023.
[11] T. Fang, Y. Zhang, Y. Yang, and C. Wang, “Prompt
tuning for graph neural networks,” arXiv preprint, 2022.
[12] T. Fang, Y. Zhang, Y. Yang, C. Wang, and L. Chen,
“Universal Prompt Tuning for Graph Neural Networks,”
in NeurIPS, 2023.
[13] X. Fang, L. Liu, J. Lei, D. He, S. Zhang, J. Zhou, F.Wang,
H. Wu, and H. Wang, “Geometry-enhanced molecular
representation learning for property prediction,”
Nature Machine Intelligence, vol. 4, no. 2, pp. 127–134,
2022.
[14] B. Fatemi, J. Halcrow, and B. Perozzi, “Talk like a
graph: Encoding graphs for large language models,”
arXiv preprint, 2023.
[15] C. Finn, P. Abbeel, and S. Levine, “Model-Agnostic
Meta-Learning for Fast Adaptation of Deep Networks,”
in ICML, 2017.
[16] T. Gao, A. Fisch, and D. Chen, “Making Pre-Trained
Language Models Better Few-Shot Learners,” in ACL,
2021, pp. 3816–3830.
[17] Q. Ge, Z. Zhao, Y. Liu, A. Cheng, X. Li, S. Wang,
and D. Yin, “Enhancing Graph Neural Networks with
Structure-Based Prompt,” arXiv preprint, 2023.
[18] C. Gong, X. Li, J. Yu, C. Yao, J. Tan, C. Yu, and D. Yin,
“Prompt Tuning for Multi-View Graph Contrastive
Learning,” arXiv preprint, 2023.
[19] A. Grover and J. Leskovec, “node2vec: Scalable feature
learning for networks,” in KDD, 2016, pp. 855–864.
[20] Y. Guo, C. Yang, Y. Chen, J. Liu, C. Shi, and J. Du,
“A Data-Centric Framework to Endow Graph Neural
Networks with Out-of-Distribution Detection Ability,”
in KDD, 2023, pp. 638–648.
[21] W. Hamilton, Z. Ying, and J. Leskovec, “Inductive
Representation Learning on Large Graphs,” in NeurIPS,
vol. 30, 2017.
[22] B. Hao, C. Yang, L. Guo, J. Yu, and H. Yin, “Motif-
Based Prompt Learning for Universal Cross-Domain
Recommendation,” in WSDM, 2024.
[23] A. Hasanzadeh, E. Hajiramezanali, K. Narayanan,
N. Duffield, M. Zhou, and X. Qian, “Semi-Implicit
Graph Variational Auto-Encoders,” in NeurIPS, vol. 32,
2019.
[24] K. Hassani and A. H. K. Ahmadi, “Contrastive Multi-
View Representation Learning on Graphs,” in ICML,
vol. 119, 2020, pp. 4116–4126.
[25] A. Haviv, J. Berant, and A. Globerson, “BERTese:
Learning to Speak to BERT,” in EACL, 2021, pp. 3618–
3623.
[26] M. He, Z. Wei, z. Huang, and H. Xu, “BernNet:
Learning Arbitrary Graph Spectral Filters via Bernstein
Approximation,” in NeurIPS, vol. 34, 2021, pp. 14 239–
14 251.
[27] Z. Hou, X. Liu, Y. Cen, Y. Dong, H. Yang, C. Wang, and
J. Tang, “GraphMAE: Self-Supervised Masked Graph
Autoencoders,” in KDD, 2022, pp. 594–604.
[28] Z. Hou, Y. He, Y. Cen, X. Liu, Y. Dong, E. Kharlamov,
and J. Tang, “GraphMAE2: A Decoding-Enhanced
Masked Self-Supervised Graph Learner,” in The Web
Conference, 2023, pp. 737–746.
[29] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang,
L. Wang, and W. Chen, “LoRA: Low-Rank Adaptation
of Large Language Models,” arXiv preprint, 2021.
[30] W. Hu, B. Liu, J. Gomes, M. Zitnik, P. Liang, V. S.
Pande, and J. Leskovec, “Strategies for Pre-training
Graph Neural Networks,” in ICLR, 2020.
[31] Z. Hu, Y. Dong, K. Wang, K.-W. Chang, and Y. Sun,
“GPT-GNN: Generative Pre-Training of Graph Neural
Networks,” in KDD, 2020, pp. 1857–1867.
[32] Q. Huang, H. Ren, P. Chen, G. Krˇzmanc, D. Zeng,
P. Liang, and J. Leskovec, “PRODIGY: Enabling Incontext
Learning Over Graphs,” in NeurIPS, 2023.
[33] X. Jiang, T. Jia, Y. Fang, C. Shi, Z. Lin, and H. Wang,
“Pre-training on Large-Scale Heterogeneous Graph,” in
KDD, 2021, pp. 756–766.
[34] X. Jiang, Y. Lu, Y. Fang, and C. Shi, “Contrastive
Pre-Training of GNNs on Heterogeneous Graphs,” in
CIKM, 2021, pp. 803–812.
[35] Z. Jiang, F. F. Xu, J. Araki, and G. Neubig, “How
Can We Know What Language Models Know?” TACL,
vol. 8, pp. 423–438, 2020.
[36] B. Jin, W. Zhang, Y. Zhang, Y. Meng, X. Zhang, Q. Zhu,
and J. Han, “Patton: Language Model Pretraining on
Text-Rich Networks,” in ACL, 2023.

and J. Han, “Patton: Language Model Pretraining on
Text-Rich Networks,” in ACL, 2023.
[37] M. Jin, Y. Zheng, Y.-F. Li, C. Gong, C. Zhou, and S. Pan,
“Multi-Scale Contrastive Siamese Networks for Self-
Supervised Graph Representation Learning,” in IJCAI,
2021, pp. 1477–1483.
[38] W. Jin, T. Derr, Y. Wang, Y. Ma, Z. Liu, and J. Tang,
“Node Similarity Preserving Graph Convolutional Networks,”
in WSDM, 2021, pp. 148–156.
[39] D. Kim and A. Oh, “How to Find Your Friendly
Neighborhood: Graph Attention Design with Self-
Supervision,” in ICLR, 2021.
[40] B. Lester, R. Al-Rfou, and N. Constant, “The Power
of Scale for Parameter-Efficient Prompt Tuning,” in
EMNLP, 2021, pp. 3045–3059.
[41] J. Li, R. Wu, W. Sun, L. Chen, S. Tian, L. Zhu, C. Meng,
Z. Zheng, and W. Wang, “What’s Behind the Mask:
Understanding Masked Graph Modeling for Graph
Autoencoders,” in KDD, 2023, pp. 1268–1279.
[42] S. Li, X. Han, and J. Bai, “AdapterGNN: Efficient
Delta Tuning Improves Generalization Ability in Graph
Neural Networks,” arXiv preprint, 2023.
[43] X. L. Li and P. Liang, “Prefix-Tuning: Optimizing
Continuous Prompts for Generation,” in ACL-IJCNLP,
2021, pp. 4582–4597.
[44] X. Li, D. Lian, Z. Lu, J. Bai, Z. Chen, and X. Wang,
“GraphAdapter: Tuning Vision-Language Models With
Dual Knowledge Graph,” in NeurIPS, 2023.
[45] Y. Li and B. Hooi, “Prompt-Based Zero-and Few-Shot
Node Classification: A Multimodal Approach,” arXiv
preprint, 2023.
[46] Y. Li, Z. Li, P. Wang, J. Li, X. Sun, H. Cheng, and J. X.
Yu, “A survey of graph meets large language model:
Progress and future directions,” arXiv preprint, 2023.
[47] H. Liu, J. Feng, L. Kong, N. Liang, D. Tao, Y. Chen, and
M. Zhang, “One for All: Towards Training One Graph
Model for All Classification Tasks,” arXiv preprint, 2023.
[48] J. Liu, C. Yang, Z. Lu, J. Chen, Y. Li, M. Zhang, T. Bai,
Y. Fang, L. Sun, P. S. Yu, and C. Shi, “Towards Graph
Foundation Models: A Survey and Beyond,” arXiv
preprint, 2023.
[49] P. Liu, Y. Ren, and Z. Ren, “GIT-Mol: A Multi-modal
Large Language Model for Molecular Science with
Graph, Image, and Text,” arXiv preprint, 2023.
[50] P. Liu,W. Yuan, J. Fu, Z. Jiang, H. Hayashi, and G. Neubig,
“Pre-train, Prompt, and Predict: A Systematic
Survey of Prompting Methods in Natural Language
Processing,” ACM Computing Surveys, vol. 55, no. 9, pp.
195:1–195:35, 2023.
[51] X. Liu, K. Ji, Y. Fu, W. Tam, Z. Du, Z. Yang, and J. Tang,
“P-Tuning: Prompt Tuning Can Be Comparable to Fine-
Tuning Across Scales and Tasks,” in ACL, vol. 2, 2022,
pp. 61–68.
[52] Z. Liu, X. Yu, Y. Fang, and X. Zhang, “Graphprompt:
Unifying Pre-Training and Downstream Tasks for
Graph Neural Networks,” in The Web Conference, 2023,
pp. 417–428.
[53] Z. Liu, S. Li, Y. Luo, H. Fei, Y. Cao, K. Kawaguchi,
X. Wang, and T.-S. Chua, “MolCA: Molecular Graph-
Language Modeling with Cross-Modal Projector and
Uni-Modal Adapter,” in EMNLP, 2023.
[54] Y. Long, M. Wu, Y. Liu, Y. Fang, C. K. Kwoh, J. Chen,
J. Luo, and X. Li, “Pre-training graph neural networks
for link prediction in biomedical networks,” Bioinformatics,
vol. 38, no. 8, pp. 2254–2262, 2022.
[55] Y. Ma, N. Yan, J. Li, M. Mortazavi, and N. V. Chawla,
“HetGPT: Harnessing the Power of Prompt Tuning in
Pre-Trained Heterogeneous Graph Neural Networks,”
arXiv preprint, 2023.
[56] M. Niepert, M. Ahmed, and K. Kutzkov, “Learning
Convolutional Neural Networks for Graphs,” in ICML,
2016, pp. 2014–2023.
[57] M. Ou, P. Cui, J. Pei, Z. Zhang, and W. Zhu, “Asymmetric
Transitivity Preserving Graph Embedding,” in
KDD, 2016, pp. 1105–1114.
[58] S. Pan, R. Hu, G. Long, J. Jiang, L. Yao, and C. Zhang,
“Adversarially Regularized Graph Autoencoder for
Graph Embedding,” in IJCAI, 2018, pp. 2609–2615.
[59] S. Pan, L. Luo, Y. Wang, C. Chen, J. Wang, and X. Wu,
“Unifying Large Language Models and Knowledge
Graphs: A Roadmap,” arXiv preprint, 2023.
[60] J. Park, A. Patel, O. Z. Khan, H. J. Kim, and J.-K. Kim,
“Graph-Guided Reasoning for Multi-Hop Question
Answering in Large Language Models,” arXiv preprint,
2023.
[61] J. Park, M. Lee, H. J. Chang, K. Lee, and J. Y. Choi,
“Symmetric Graph Convolutional Autoencoder for
Unsupervised Graph Representation Learning,” in
ICCV, 2019, pp. 6519–6528.
[62] Z. Peng, W. Huang, M. Luo, Q. Zheng, Y. Rong, T. Xu,
and J. Huang, “Graph Representation Learning via
Graphical Mutual Information Maximization,” in The
Web Conference, 2020, pp. 259–270.
[63] B. Perozzi, R. Al-Rfou, and S. Skiena, “Deepwalk:
Online learning of social representations,” in KDD,
2014, pp. 701–710.
[64] C. Qian, H. Tang, Z. Yang, H. Liang, and Y. Liu, “Can
Large Language Models Empower Molecular Property
Prediction?” arXiv preprint, 2023.
[65] G. Qin and J. Eisner, “Learning How to Ask: Querying
LMs with Mixtures of Soft Prompts,” in NAACL-HLT,
2021, pp. 5203–5212.
[66] J. Qiu, Q. Chen, Y. Dong, J. Zhang, H. Yang, M. Ding,
K. Wang, and J. Tang, “GCC: Graph Contrastive
Coding for Graph Neural Network Pre-Training,” in
KDD, 2020, pp. 1150–1160.
[67] J. Robinson, C. M. Rytting, and D. Wingate, “Leveraging
Large Language Models for Multiple Choice
Question Answering,” arXiv preprint, 2023.
[68] Y. Rong, Y. Bian, T. Xu, W. Xie, Y. WEI, W. Huang,
and J. Huang, “Self-Supervised Graph Transformer on
Large-Scale Molecular Data,” in NeurIPS, vol. 33, 2020,
pp. 12 559–12 571.
[69] M. T. Rosenstein, Z. Marx, L. P. Kaelbling, and T. G.
Dietterich, “To transfer or not to transfer,” in NeurIPS,
vol. 898, 2005.
[70] T. Schick and H. Sch ¨ utze, “Few-Shot Text Generation
with Natural Language Instructions,” in EMNLP, 2021,
pp. 390–402.
[71] ——, “It’s Not Just Size That Matters: Small Language
Models Are Also Few-Shot Learners,” in NAACL-HLT,
2021, pp. 2339–2352.

[72] Y. Shi, Z. Huang, S. Feng, H. Zhong, W. Wang, and
Y. Sun, “Masked Label Prediction: Unified Message
Passing Model for Semi-Supervised Classification,” in
IJCAI, vol. 2, 2021, pp. 1548–1554.
[73] T. Shin, Y. Razeghi, R. L. L. IV, E. Wallace, and S. Singh,
“Autoprompt: Eliciting Knowledge from Language
Models with Automatically Generated Prompts,” in
EMNLP, 2020, pp. 4222–4235.
[74] R. Shirkavand and H. Huang, “Deep Prompt Tuning
for Graph Transformers,” arXiv preprint, 2023.
[75] A. Subramonian, “MOTIF-Driven Contrastive Learning
of Graph Representations,” in AAAI, vol. 35, 2021,
pp. 15 980–15 981.
[76] F.-Y. Sun, J. Hoffmann, V. Verma, and J. Tang, “InfoGraph:
Unsupervised and Semi-supervised Graph-
Level Representation Learning via Mutual Information
Maximization,” in ICLR, 2020.
[77] M. Sun, J. Xing, H.Wang, B. Chen, and J. Zhou, “MoCL:
Data-driven Molecular Fingerprint via Knowledgeaware
Contrastive Learning from Molecular Graph,”
in KDD, 2021, pp. 3585–3594.
[78] M. Sun, K. Zhou, X. He, Y. Wang, and X. Wang, “GPPT:
Graph Pre-Training and Prompt Tuning to Generalize
Graph Neural Networks,” in KDD, 2022, pp. 1717–
1727.
[79] X. Sun, H. Yin, B. Liu, H. Chen, J. Cao, Y. Shao,
and N. Q. Viet Hung, “Heterogeneous Hypergraph
Embedding for Graph Classification,” in WSDM, 2021,
pp. 725–733.
[80] X. Sun, H. Cheng, J. Li, B. Liu, and J. Guan, “All in One:
Multi-Task Prompting for Graph Neural Networks,” in
KDD, 2023, pp. 2120–2131.
[81] S. Suresh, P. Li, C. Hao, and J. Neville, “Adversarial
Graph Augmentation to Improve Graph Contrastive
Learning,” in NeurIPS, vol. 34, 2021, pp. 15 920–15 933.
[82] Q. Tan, N. Liu, X. Huang, S.-H. Choi, L. Li, R. Chen, and
X. Hu, “S2GAE: Self-Supervised Graph Autoencoders
are Generalizable Learners with Graph Masking,” in
WSDM, 2023, pp. 787–795.
[83] Z. Tan, R. Guo, K. Ding, and H. Liu, “Virtual Node
Tuning for Few-shot Node Classification,” in KDD,
2023, pp. 2177–2188.
[84] S. Thakoor, C. Tallec, M. G. Azar, R. Munos,
P. Veliˇckovi´c, and M. Valko, “Bootstrapped representation
learning on graphs,” in ICLR, 2021.
[85] Y. Tian, H. Song, Z. Wang, H. Wang, Z. Hu, F. Wang,
N. V. Chawla, and P. Xu, “Graph Neural Prompting
with Large Language Models,” arXiv preprint, 2023.
[86] M. Tsimpoukelli, J. L. Menick, S. Cabi, S. M. A.
Eslami, O. Vinyals, and F. Hill, “Multimodal Few-Shot
Learning with Frozen Language Models,” in NeurIPS,
vol. 34, 2021, pp. 200–212.
[87] P. Velickovic, G. Cucurull, A. Casanova, A. Romero,
P. Li` o, and Y. Bengio, “Graph Attention Networks,” in
ICLR, 2018.
[88] P. Velickovic, W. Fedus, W. L. Hamilton, P. Li` o, Y. Bengio,
and R. D. Hjelm, “Deep Graph Infomax,” in ICLR,
2019.
[89] C. Wang, S. Pan, G. Long, X. Zhu, and J. Jiang,
“MGAE: Marginalized Graph Autoencoder for Graph
Clustering,” in CIKM, 2017, pp. 889–898.
[90] H. Wang, T. Fu, Y. Du, W. Gao, K. Huang, Z. Liu et al.,
“Scientific discovery in the age of artificial intelligence,”
Nature, vol. 620, no. 7972, pp. 47–60, 2023.
[91] J. Wang, D. Chen, C. Luo, X. Dai, L. Yuan, Z. Wu, and
Y.-G. Jiang, “ChatVideo: A Tracklet-centric Multimodal
and Versatile Video Understanding System,” arXiv
preprint, 2023.
[92] P. Wang, K. Agarwal, C. Ham, S. Choudhury, and
C. K. Reddy, “Self-Supervised Learning of Contextual
Embeddings for Link Prediction in Heterogeneous
Networks,” in The Web Conference, 2021, pp. 2946–2957.
[93] R. Wang, Y. Li, S. Lin, W. Wu, H. Xie, Y. Xu, and J. C.
Lui, “Common neighbors matter: fast random walk
sampling with common neighbor awareness,” TKDE,
vol. 35, no. 5, pp. 4570–4584, 2022.
[94] X. Wang, Y. Zhang, and C. Shi, “Hyperbolic heterogeneous
information network embedding,” in AAAI,
vol. 33, 2019, pp. 5337–5344.
[95] X. Wang, N. Liu, H. Han, and C. Shi, “Self-supervised
Heterogeneous Graph Neural Network with Cocontrastive
Learning,” in KDD, 2021, pp. 1726–1736.
[96] Y. Wang, N. Lipka, Ryan A. Rossi, Alexa Siu, Ruiyi
Zhang, and Tyler Derr, “Knowledge Graph Prompting
for Multi-Document Question Answering,” arXiv
preprint, 2023.
[97] Z. Wen and Y. Fang, “Augmenting Low-Resource Text
Classification with Graph-Grounded Pre-Training and
Prompting,” in SIGIR, 2023, pp. 506–516.
[98] ——, “Prompt Tuning on Graph-augmented Lowresource
Text Classification,” arXiv preprint, 2023.
[99] Z. Wen, Y. Fang, Y. Liu, Y. Guo, and S. Hao, “Voucher
Abuse Detection with Prompt-based Fine-tuning on
Graph Neural Networks,” arXiv preprint, 2023.
[100] J. Wu, X. Wang, F. Feng, X. He, L. Chen, J. Lian, and
X. Xie, “Self-supervised Graph Learning for Recommendation,”
in SIGIR, 2021.
[101] J.Wu, S. Li, A. Deng, M. Xiong, and H. Bryan, “Promptand-
Align: Prompt-Based Social Alignment for Few-
Shot Fake News Detection,” in CIKM, 2023.
[102] X. Wu, K. Zhou, M. Sun, X. Wang, and N. Liu, “A
Survey of Graph Prompting Methods: Techniques,
Applications, and Challenges,” arXiv preprint, 2023.
[103] Y. Wu, R. Xie, Y. Zhu, F. Zhuang, X. Zhang, L. Lin, and
Q. He, “Personalized Prompt for Sequential Recommendation,”
arXiv preprint, 2023.
[104] Y. Xie, Z. Xu, and S. Ji, “Self-Supervised Representation
Learning via Latent Graph Prediction,” in ICML, 2022,
pp. 24 460–24 477.
[105] Y. Xie, Z. Xu, J. Zhang, Z. Wang, and S. Ji, “Self-
Supervised Learning of Graph Neural Networks: A
Unified Review,” TPAML, vol. 45, no. 2, pp. 2412–2429,
2023.
[106] D. Xu, W. Cheng, D. Luo, H. Chen, and X. Zhang, “InfoGCL:
Information-Aware Graph Contrastive Learning,”
in NeurIPS, vol. 34, 2021, pp. 30 414–30 425.
[107] K. Xu, W. Hu, J. Leskovec, and S. Jegelka, “How
Powerful are Graph Neural Networks?” in ICLR, 2018.
[108] C. Yang, D. Bo, J. Liu, Y. Peng, B. Chen, H. Dai et al.,
“Data-centric graph learning: A survey,” arXiv preprint,
2023.
[109] H. Yang, X. Zhao, Y. Li, H. Chen, and G. Xu, “An
Empirical Study Towards Prompt-Tuning for Graph
Contrastive Pre-Training in Recommendations,” in
NeurIPS, 2023.
[110] Z. Yi, I. Ounis, and C. Macdonald, “Contrastive Graph
Prompt-tuning for Cross-domain Recommendation,”
TOIS, 2023.
[111] Y. You, T. Chen, Y. Sui, T. Chen, Z. Wang, and Y. Shen,
“Graph contrastive learning with augmentations,” in
NeurIPS, 2020, pp. 5812–5823.
[112] J. Yu, H. Yin, X. Xia, T. Chen, L. Cui, and N. Q. V. Hung,
“Are Graph Augmentations Necessary? Simple Graph
Contrastive Learning for Recommendation,” in SIGIR,
2022.
[113] J. Yu, H. Yin, X. Xia, T. Chen, J. Li, and Z. Huang,
“Self-Supervised Learning for Recommender Systems:
A Survey,” TKDE, pp. 1–20, 2023.
[114] H. Zhang, X. Li, and L. Bing, “Video-LLaMA: An
Instruction-tuned Audio-Visual Language Model for
Video Understanding,” arXiv preprint, 2023.
[115] J. Zhang, H. Zhang, C. Xia, and L. Sun, “Graph-
Bert: Only Attention is Needed for Learning Graph
Representations,” arXiv preprint, 2020.
[116] J. Zhang, Y. Dong, Y. Wang, J. Tang, and M. Ding,
“ProNE: Fast and Scalable Network Representation
Learning,” in IJCAI, 2019, pp. 4278–4284.
[117] M. Zhang and Y. Chen, “Link Prediction Based on
Graph Neural Networks,” in NeurIPS, 2018.
[118] T. Zhang, F. Ladhak, E. Durmus, P. Liang, K. McKeown,
and T. B. Hashimoto, “Benchmarking Large Language
Models for News Summarization,” arXiv preprint, 2023.
[119] W. Zhang, Y. Zhu, M. Chen, Y. Geng, Y. Huang, Y. Xu,
W. Song, and H. Chen, “Structure Pretraining and
Prompt Tuning for Knowledge Graph Transfer,” in The
Web Conference, 2023, pp. 2581–2590.
[120] Z. Zhang, H. Li, Z. Zhang, Y. Qin, X.Wang, andW. Zhu,
“Large graph models: A perspective,” arXiv preprint,
2023.
[121] H. Zhao, S. Liu, C. Ma, H. Xu, J. Fu, Z.-H. Deng,
L. Kong, and Q. Liu, “GIMLET: A Unified Graph-
Text Model for Instruction-Based Molecule Zero-Shot
Learning,” arXiv preprint, 2023.
[122] W. Zhao, Q. Wu, C. Yang, and J. Yan, “GraphGLOW:
Universal and Generalizable Structure Learning for
Graph Neural Networks,” in KDD, 2023, pp. 3525–
3536.
[123] Y. Zhu, Y. Xu, F. Yu, Q. Liu, S.Wu, and L.Wang, “Graph
Contrastive Learning with Adaptive Augmentation,”
in The Web Conference, 2021, pp. 2069–2080.
[124] Y. Zhu, J. Guo, and S. Tang, “SGL-PT: A Strong Graph
Learner with Graph Prompt Tuning,” arXiv preprint,
2023.
[125] Y. Zhu, Y.Wang, H. Shi, Z. Zhang, and S. Tang, “Graph-
Control: Adding Conditional Control to Universal
Graph Pre-trained Models for Graph Domain Transfer
Learning,” arXiv preprint, 2023.
