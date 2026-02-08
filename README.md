# Named Entity Recognition for Biomedical Text Mining

A comprehensive comparison of state-of-the-art NER models for extracting diseases and chemicals from clinical literature using the BC5CDR corpus.

## 📋 Overview

This project implements and evaluates four different Named Entity Recognition (NER) models for identifying chemical and disease entities in biomedical texts:

- **BioBERT** - Domain-specific transformer model
- **DeBERTa-v3-base** - General-purpose transformer with advanced architecture
- **BiLSTM-CRF with BioWordVec** - Classical architecture with biomedical embeddings
- **BiLSTM-CRF with FastText** - Classical architecture with general-purpose embeddings

## 🎯 Key Results

| Model | Overall F1 | Chemical F1 | Disease F1 |
|-------|-----------|-------------|------------|
| **DeBERTa-v3-base** | **89.22%** | 92.39% | 85.43% |
| **BioBERT** | **88.42%** | 92.50% | 83.53% |
| BiLSTM-CRF (BioWordVec) | 73.75% | 73.70% | 73.80% |
| BiLSTM-CRF (FastText) | 73.34% | 74.13% | 72.33% |

## 🔬 Problem Statement

The exponential growth of biomedical literature creates significant challenges for extracting actionable information. This project addresses:

- Automated identification of chemical and disease entities in clinical texts
- Handling of specialized medical terminology with morphological variations
- Processing of abbreviations, acronyms, and evolving medical vocabulary
- Enabling downstream applications: drug discovery, adverse event detection, clinical decision support

## 📊 Dataset

**BC5CDR Corpus** (BioCreative V Chemical Disease Relation)
- 1,500 PubMed articles manually annotated by domain experts
- Training: 10,456 sentences
- Validation: 10,660 sentences
- Test: 11,730 sentences
- Entity classes: Chemical (5,385 instances) and Disease (4,424 instances)
- Annotation scheme: BIO tagging

Dataset available on [HuggingFace](https://huggingface.co/datasets/tner/bc5cdr)

## 🏗️ Architecture

### Transformer Models

#### BioBERT
- Base: BERT-base architecture (110M parameters)
- Pre-trained on 4.5B words from biomedical literature
- Hidden size: 768, Attention heads: 12, Layers: 12
- Vocabulary: 28,996 WordPiece tokens (cased)

#### DeBERTa-v3-base
- 186M parameters, 12 transformer layers
- Key innovations:
  - Disentangled attention mechanism (separate content & position embeddings)
  - Enhanced masked decoder with absolute position processing
  - Relative position encoding in attention

### Classical Models

#### BiLSTM-CRF Architecture
- Bidirectional LSTM layers (256 hidden dimensions)
- Conditional Random Field (CRF) output layer
- Two embedding variants:
  - **BioWordVec**: 200-dim, trained on PubMed + MIMIC-III (74% vocabulary coverage)
  - **FastText**: 300-dim, trained on Wikipedia + Common Crawl (85.66% coverage)

## 🔧 Preprocessing

### Transformer Models
1. Subword tokenization alignment with BIO labels
2. First subword receives original label
3. Subsequent B-tagged subwords → I-tag
4. Special tokens ([CLS], [SEP], [PAD]) → -100 label

### BiLSTM-CRF Models
1. Lowercase conversion for generalization
2. Vocabulary construction from training set
3. OOV tokens mapped to `<UNK>`

## 📈 Training Configuration

### Transformer Models
- Learning rate: 2e-5
- Batch size: 16
- Max epochs: 10
- Early stopping patience: 3
- Weight decay: 0.01
- Loss: Cross-entropy

### BiLSTM-CRF Models
- Optimizer: Adam
- Learning rate: 0.01
- Batch size: 32
- Max epochs: 10
- Gradient clipping: max norm = 1.0
- Early stopping patience: 3

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/biomedical-ner.git
cd biomedical-ner

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 💻 Usage

### Training Models

```python
# BioBERT
python train_biobert.py --model_name dmis-lab/biobert-base-cased-v1.1 \
                        --learning_rate 2e-5 \
                        --batch_size 16 \
                        --epochs 10

# DeBERTa-v3-base
python train_deberta.py --model_name microsoft/deberta-v3-base \
                        --learning_rate 2e-5 \
                        --batch_size 16 \
                        --epochs 10

# BiLSTM-CRF with BioWordVec
python train_bilstm.py --embedding biowordvec \
                       --hidden_dim 256 \
                       --learning_rate 0.01

# BiLSTM-CRF with FastText
python train_bilstm.py --embedding fasttext \
                       --hidden_dim 256 \
                       --learning_rate 0.01
```

### Evaluation

```python
# Evaluate trained model
python evaluate.py --model_path checkpoints/biobert_best.pt \
                   --test_data data/bc5cdr/test.json \
                   --reconciliation_strategy prioritized_voting
```

## 📝 Key Findings

### 1. Transformer Superiority
- Contextual embeddings and attention mechanisms provide 15+ percentage point advantage over static embeddings
- DeBERTa-v3-base achieved highest performance (89.22% F1)

### 2. Domain Pre-training Trade-offs
- General-purpose DeBERTa slightly outperformed domain-specific BioBERT
- Architectural innovations can compensate for lack of specialized pre-training
- Domain knowledge remains valuable for specific entity types

### 3. Embedding Strategy Impact
- FastText's subword awareness benefits chemical nomenclature (+0.43% F1)
- BioWordVec excels at disease detection due to clinical context exposure
- Vocabulary coverage crucial: FastText 85.66% vs BioWordVec 74%

### 4. Error Analysis
- **Transformers**: Tend to over-predict entity boundaries (e.g., including modifiers)
- **BiLSTM-CRF**: Conservative predictions with high precision but lower recall
- **Challenge areas**: Rare entities, multi-word entities, context-dependent disambiguation

## 🛠️ Methodological Contributions

1. **Prioritized Voting Strategy**: Novel reconciliation approach for subword-to-word alignment
2. **Span-level Evaluation**: Fair comparison between different tokenization schemes
3. **Comprehensive Error Analysis**: Distinct failure mode identification across architectures

## 👥 Authors

- Nagraj Venkatesham Vallakati
- Lokeshwar Reddy Peddarangareddy
- Swet Vimalkumar Patel

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
