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

In recent years, Graph Neural Networks (GNNs) have emerged as a focal point of research interest, largely attributed to their proficiency in adeptly capturing intricate relationships embedded within graph-formulated data. The landscape of GNN methodologies has seen a proliferation of diverse models, notably encompassing Graph Convolutional Networks (GCNs) [9], Graph Attention Networks (GATs) [44], and GraphSAGE [17]. These architectures have found broad application across a spectrum of tasks, spanning from node classification [27], graph classification [42], to link prediction [23], illustrating their versatility and effectiveness in handling graph-based problems.

The burgeoning field of graph representation learning, encompassing early graph embedding techniques [12, 33, 41] and the recent proliferation of Graph Neural Networks (GNNs) [13, 20, 43, 50], has unlocked significant potential for a wide array of downstream tasks, both at the node and graph levels. It's noteworthy that crafting representations at the graph level necessitates an additional step known as the ReadOut operation. This operation consolidates a graph's global information by amassing the representations of individual nodes using either flat [8, 11, 50, 56] or hierarchical [10, 21, 31, 51] pooling strategies, enabling a holistic understanding of graph structure and function.

The adoption of the pre-training and fine-tuning framework in GNNs facilitates the training of models on large-scale graph datasets, with or without labels, before fine-tuning for specific downstream tasks. This two-step process significantly improves model initialization, providing broader optima and enhanced generalization capabilities compared to training models from scratch, thus addressing the limitations posed by scarce labeled data and enhancing model performance.

There are three main categories of pre-training strategies for GNNs: node-level, edge-level, and graph-level. Node-level Pre-training emphasizes local structure representations, using contrastive learning and predictive models to maximize Mutual Information between original and perturbed views. However, it struggles to capture higher-order information, focusing mainly on local topology. Edge-level Pre-training Aimed at enhancing link prediction tasks, these strategies utilize contrastive learning to predict edge presence and adjacency matrix reconstruction. While strong in structural understanding, they overlook node properties, limiting effectiveness in graph-level tasks. Graph-level Pre-training focus on graph-level representations for subgraph tasks, this approach includes graph reconstruction and contrastive methods to encode global information. Challenges arise when transferring this knowledge to tasks with significant structural differences, potentially leading to negative transfer.

There are three primary approaches to graph prompt tuning: Meta-Learning Based Approaches: These methods utilize meta-learning techniques, such as Model-Agnostic Meta-Learning (MAML), to learn a robust starting point for prompt parameters. This enables the learned graph prompt to be adaptable across a wide range of downstream tasks. Task-Specific Tuning: This approach focuses on tuning the graph prompt specifically for individual tasks. For instance, the Graph Prompt Tuning (GPF) method [11] targets graph classification tasks. It achieves this by optimizing the prompt token and task head to maximize the likelihood of accurately predicting graph labels. Tuning in Line with Pretext: Aligning prompt tuning with the pre-training task by adopting similar loss functions or objectives, such as raphPrompt [52], GPPT [78] using the same loss function as the downstream task or employing node-level contrastive loss.

Recently, the task of novel class discovery has been proposed, which aims to identify novel classes in unlabeled data, similar to our open set event detection task. Unlike traditional unsupervised learning methods, this task leverages known data to facilitate the discovery of new classes. Existing methods typically employ a two-stage approach: (1) initializing a model using labeled data, and (2) fine-tuning the model through unsupervised clustering or pairwise determination on unlabeled data. For instance, authors in [19] proposed a constrained clustering network to recognize open set images, which first measures pairwise similarities between images using a classification model trained on labeled data, and then applies a clustering model to unlabeled data using these pairwise predictions. Similarly, authors in [17] use rank statistics to estimate pairwise similarity directly. Although these methods, which use pseudo-labels to adapt the model to unlabeled data, have achieved promising results, they overlook the noise present in the obtained pseudo-labels, making the training process unreliable. Furthermore, they fail to provide a strategy for selecting samples. Given the large amount of unlabeled data, selecting a small number of samples to improve performance has become a critical issue.

