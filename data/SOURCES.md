# Sources and scope

## Purchase data
Online Shoppers Purchasing Intention Dataset, UCI Machine Learning Repository, dataset 468.
https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset
Creators: C. O. Sakar and Y. Kastro (2018). License: Creative Commons Attribution 4.0 International (CC BY 4.0).
https://creativecommons.org/licenses/by/4.0/
The unchanged CSV is redistributed with attribution. Training uses a seeded stratified 80/20 split. PageValues is excluded because its definition incorporates transaction value and can compromise deployment relevance. Session summaries are retrospective; performance is not evidence of early-session prediction or causal effects.

## Catalog
The 30 sample product names, descriptions, and illustrative INR prices were authored for this project. They do not represent a live retailer, stock availability, or real market prices. TF-IDF similarity is not evaluated recommendation accuracy or personalized collaborative filtering.

## Sentiment
VADER: Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text. ICWSM.
https://github.com/cjhutto/vaderSentiment
Package/lexicon: MIT license. VADER handles common negation and intensifiers; English sarcasm, context, and mixed reviews remain limitations. No custom review benchmark accuracy is claimed.

## Image model
Torchvision MobileNet V3 Small, IMAGENET1K_V1 pretrained weights.
https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_small.html
https://github.com/pytorch/vision
Torchvision code: BSD 3-Clause. Pretrained weights and source training datasets may have separate terms; consult upstream documentation before commercial reuse. Weights are downloaded from the official PyTorch URL, not represented as a newly trained model. Predictions use the original 1,000 ImageNet classes. There is no claimed evaluation on this project's product uploads and no custom four-class product taxonomy.
