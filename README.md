# S2S_MT

presentation of the project and what we do/ want to do

https://huggingface.co/learn/audio-course/chapter7/speech-to-speech

we can try STT + MT + TTS  and just STTT + TTS  (recently some papers focused on S2S without an intermediate passage to text)

maybe we can try on one "common" language and one "rare" (with less data available) and compare the results

# Models
which models we used/tried (why?)

# Datasets
which datasets (why?)

https://arxiv.org/abs/2211.04508 (https://github.com/facebookresearch/fairseq/tree/ust/examples/speech_matrix)
https://aclanthology.org/N19-1202.pdf (should be here : https://ict.fbk.eu/must-c/   but I don't find)
https://arxiv.org/abs/2201.03713 (https://github.com/google-research-datasets/cvss)
https://arxiv.org/abs/2204.10593 (https://github.com/pedrodke/libris2s)

# Evaluation
how we evaluate and the results

https://arxiv.org/pdf/2110.13877.pdf

Reference Text: ASR Transcription with MT Metrics

For applications combining translation with synthesis such as speech-to-speech (S2S) or text-to-speech translation (T2S), previous work has exclusively transcribed synthesized speech with ASR to evaluate with the text-based metric BLEU [8], [11], [12], in part due to the absence of datasets with parallel speech.

To evaluate synthesized speech translations with standard automatic MT metrics, previous work on neural speech-to-speech translation [8], [11], [12] has utilized large ASR models trained on hundreds of hours of external corpora in the target language or commercial transcription services to transcribe synthesized samples for comparison against text references. The use of high-quality external models is to prevent the introduction of ASR errors which may impact the downstream MT metric.

Previous work has evaluated using ASR and BLEU only [14] and have experiments with high-resource languages with standardized orthographies only; however, language dialects often have non-standardized orthographies which we show disproportionately affect word-level metrics like BLEU. With this in mind, we also compare two character-level MT metrics. chrF [15] computes F1-score of character n-grams, while character-level BLEU (charBLEU) computes BLEU on character rather than word sequences. We use SacreBLEU [16] to calculate both BLEU and chrF scores.

-> get transcription with already pretrained models of outputs and predicted outputs and compare them with BLEU, chrF and charBLEU.


Reference Speech: Mel-Cepstral Distortion (MCD)

https://github.com/chenqi008/pymcd

-> metric directly on audios


Try to reproduce the results of this paper ??



## Organization

organization of the code

## How to run it

libraries to install and explaination of how to run it

libraries : (maybe make a requirements file)

torch
scipy
transformers
datasets
evaluate
soundfile
librosa
SentencePiece