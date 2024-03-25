# S2S_MT

This repository contains the code referring to our project of Speech-to-Speech translation (S2S) for TPT-...
The README contains information on how to run the code, and general descriptions of the methods and datasets we used, and the results we got.

## Requirements

Recommended OS: Linux or Windows 11

### _Make a requirements file_

## Data and File Structure

## Running the Code

To run the pipeline, these are the parameters:

```
--stt_model = Abbreviated model for STT (see abbreviation table)
--mt_model = Abbreviated model for MT (see abbreviation table)
--tts_model = Abbreviated model for TTS (see abbreviation table)
--src_lan = Abbreviated name of the source language ('en' for English, 'fr' for French)
--tgt_lan = Abbreviated name of the target language ('en' for English, 'fr' for French)
--data_dir = directory containing the audio files and possibly the ground truth transcriptions
--device = which device the models on
--out_dir = filepath to save the outputs to
```

An example command to run it:

```
python main.py \
    --stt_model=fb-s2t-small \
    --mt_model=m2m \
    --tts_model=fb-tts-fra \
    --src_lan=en \
    --tgt_lan=fr \
    --data_dir=NoneForNow \
    --device=cuda:0 \
    --out_dir=out
```

## Other stuff

we can try STT + MT + TTS  and just STTT + TTS  (recently some papers focused on S2S without an intermediate passage to text)

maybe we can try on one "common" language and one "rare" (with less data available) and compare the results

## Models

which models we used/tried (why?)
list of model names:
- fb-s2t-small = facebook/s2t-small-librispeech-asr
- m2m = facebook/m2m100_418M
- fb-tts-fra = facebook/mms-tts-fra

## Dataset

-> need datasets with input audio and labels (audio and text of translation)

-> get transcription with already pretrained models of outputs and predicted outputs and compare them with BLEU, chrF and charBLEU.

CVSS [1] : presentation of dataset, organization and analysis ?

-> availalble on hugging face

https://arxiv.org/abs/2201.03713 (https://github.com/google-research-datasets/cvss)


## Evaluation

To evaluate the model, we used the metrics presented in [2]. Two types of metrics are presented; metrics based on a reference text (BLEU, charBLEU and chrF) and a metric based on a reference speech (MCD).

### Reference Text: ASR Transcription with MT Metrics

From [2] :

For applications combining translation with synthesis such as speech-to-speech (S2S) or text-to-speech translation (T2S), previous work has exclusively transcribed synthesized speech with ASR to evaluate with the text-based metric BLEU, in part due to the absence of datasets with parallel speech.

To evaluate synthesized speech translations with standard automatic MT metrics, previous work on neural speech-to-speech translation has utilized large ASR models trained on hundreds of hours of external corpora in the target language or commercial transcription services to transcribe synthesized samples for comparison against text references. The use of high-quality external models is to prevent the introduction of ASR errors which may impact the downstream MT metric.

Previous work has evaluated using ASR and BLEU only and have experiments with high-resource languages with standardized orthographies only; however, language dialects often have non-standardized orthographies which we show disproportionately affect word-level metrics like BLEU. With this in mind, we also compare two character-level MT metrics. chrF computes F1-score of character n-grams, while character-level BLEU (charBLEU) computes BLEU on character rather than word sequences. We use SacreBLEU to calculate both BLEU and chrF scores.

### Reference Speech: Mel-Cepstral Distortion (MCD)

Mel Cepstral Distortion (MCD) measures the difference between two sets of Mel Cepstral Coefficients (MCC). MCC are a set of coefficients derived from Mel-frequency cepstral analysis, a technique that captures the perceptually relevant characteristics of speech signals. MCD is computed by aligning the MCC vectors of two speech signals, typically using Dynamic Time Warping (DTW), and then calculating the Euclidean distance between corresponding pairs of aligned vectors. Lower MCD values indicate better similarity between the original and synthesized speech.

Here is how to compute MCD between two audios $y$ and $\hat{y}$ [3, 4] : 

$$MCD(y, \hat{y}) = \frac{10}{\ln(10)} \cdot \frac{1}{N} \sum_{n=0}^{N-1} \sqrt{2 \sum_{t=1}^{T} ||y_{t,n} - \hat{y}_{t,n}||}$$

-> the voice, intonnation, lentgh... affect the results. Example : two translations (same intonation from the model) 264.38 dB and the real output (with my voice and intonation) with a translation 1039.26

## References

[1] Jia, Yeting et al. “CVSS Corpus and Massively Multilingual Speech-to-Speech Translation.” International Conference on Language Resources and Evaluation (2022), https://doi.org/10.48550/arXiv.2201.03713.

[2] Salesky, Elizabeth et al. “Assessing Evaluation Metrics for Speech-to-Speech Translation.” 2021 IEEE Automatic Speech Recognition and Understanding Workshop (ASRU) (2021): 733-740, https://doi.org/10.48550/arXiv.2110.13877.

[3] Haque, Albert et al. “Conditional End-to-End Audio Transforms.” Interspeech (2018), https://doi.org/10.48550/arXiv.1804.00047.

[4] Kominek, John et al. “Synthesizer voice quality of new languages calibrated with mean mel cepstral distortion.” Workshop on Spoken Language Technologies for Under-resourced Languages (2008).