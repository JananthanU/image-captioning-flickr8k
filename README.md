# Image Captioning on Flickr8k

Implementation and evaluation of five image captioning architectures on the
[Flickr8k dataset](https://www.kaggle.com/datasets/adityajn105/flickr8k),
progressing from a simple CNN-LSTM baseline to a Transformer decoder with
cross-attention.

## Models

| Model | Description |
|---|---|
| **Show and Tell** | ResNet-18 encoder + LSTM decoder, transfer learning |
| **Show, Attend and Tell** | Additive (Bahdanau) spatial attention over CNN feature maps |
| **Dot-Product Attention** | Scaled dot-product attention as alternative to additive |
| **GloVe Embeddings** | Pretrained word vectors vs. random initialization |
| **Transformer Decoder** | Cross-attention to spatial image features |

All models are evaluated with greedy decoding and beam search (k=5) using
perplexity, BLEU-1–4, and METEOR.

## Results

| Model | Perplexity ↓ | BLEU-4 ↑ | METEOR ↑ |
|---|---|---|---|
| Show and Tell | **14.71** | 0.1863 | 0.3877 |
| Show, Attend and Tell | 16.49 | **0.1939** | **0.4021** |
| GloVe Init | 15.24 | 0.1712 | 0.3801 |
| Dot-Product Attention | 15.21 | 0.1897 | 0.3989 |
| Transformer | 15.27 | 0.1751 | 0.3842 |

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
