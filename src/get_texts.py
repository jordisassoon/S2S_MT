from datasets import load_dataset
import pandas as pd

from token_hf import token

cvss = load_dataset(
        "google/cvss", "cvss_c", languages=["fr"], split="validation", trust_remote_code=True
    )
    
# Target, French
common_voice = load_dataset(
        "mozilla-foundation/common_voice_4_0", "fr", split="validation", trust_remote_code=True, token=token, cache_dir='/data/kchardon-22/datasets'
)



source_text = [source['text'] for source in cvss]

target_text = [target['sentence'].lower() for target in common_voice]


# dictionary of lists 
dict = {'source': source_text, 'target': target_text} 
    
df = pd.DataFrame(dict)

df.to_csv('source_target_texts.csv')