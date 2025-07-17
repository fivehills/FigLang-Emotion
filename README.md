
# Affective Profiles of Figurative Language: Cognitive and Computational Insights

This repository contains the code, datasets, and analysis scripts used in the paper:

**Sun, K., Wang, R., & Wu, Y.** (2025). *Affective Profiles of Figurative Language: Cognitive and Computational Insights*. _Emotion_ (in review).

## 📌 Overview

This study investigates the emotional characteristics of four types of figurative language—**metaphors**, **similes**, **idioms**, and **sarcasm**—by combining **human-annotated datasets** with **BERT-based emotion classification models**. We reveal distinct affective tendencies for each type, offering insights into the interaction between linguistic creativity, emotion, and cognition.

## 🧠 Research Questions

- Do different types of figurative language exhibit distinct affective profiles?
- Can computational models reliably capture these emotional patterns?
- What are the implications for affective computing and cognitive semantics?

## 🔍 Datasets

The following resources are provided:

- `data/figurative_emotion_dataset.csv`: Annotated examples of figurative expressions with emotion labels (valence, arousal, emotion category).
- `data/benchmark_dataset.csv`: Standardized benchmark dataset used to fine-tune BERT models.
- `data/metadata.json`: Meta-information about annotation procedures, inter-rater agreement, and label taxonomy.

## 🧪 Code and Tools

- `src/classification/`: Scripts to fine-tune BERT models on emotional data.
- `src/analysis/`: Notebooks for statistical analysis and figure generation.
- `src/utils/`: Helper functions for preprocessing, evaluation, and visualization.
- `models/`: Pretrained checkpoints and output predictions (optional upload or link to external repository).

### Requirements

Tested on:

```bash
Python 3.10
transformers==4.30.2
scikit-learn==1.2.2
pandas==1.5.3
seaborn==0.12.2
````

Install dependencies:

```bash
pip install -r requirements.txt
```

## 🔄 Reproducibility

To reproduce the main results:

```bash
# Preprocess data
python src/utils/preprocess.py --input data/figurative_emotion_dataset.csv

# Train BERT classifier
python src/classification/train_bert.py --config configs/emotion_model.yaml

# Run analysis
jupyter notebook src/analysis/affective_profiles.ipynb
```

## 🔓 Transparency and Openness

We report all preprocessing, modeling, and analysis steps in accordance with the [APA TOP Guidelines](https://www.cos.io/initiatives/top-guidelines).
This study was **not preregistered**.
Data and code are openly available under the MIT License.

## 📁 Citation

Please cite the following if you use this code or data:

```bibtex
@article{sun2025figurative,
  title={Affective Profiles of Figurative Language: Cognitive and Computational Insights},
  author={Sun, Kun and Wang, Rong and Wu, Yun},
  journal={Emotion},
  year={2025},
  note={Manuscript under review}
}
```

## 🤝 License

This project is licensed under the [MIT License](LICENSE).

## 🙋 Contact

For questions or collaboration, please contact:

* Kun Sun: `sharpksun@hotmail.com`
* GitHub issues are welcome for reporting bugs or requesting features.


