# %%

import pandas as pd
from datasets import load_dataset
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

import unicodedata
from huggingface_hub import hf_hub_download
from pybloomfilter import BloomFilter

# %%

CWD = Path.cwd()
FIGS = CWD / "figs"
FIGS.mkdir(exist_ok=True)

hf_name = "chcaa/memo_enriched"

# %%

# get hf
ds = load_dataset(hf_name)
df = ds["train"].to_pandas()
df.head()

# %%

# get OCRQA model
known = BloomFilter.open(hf_hub_download("PascaleF/ocrqa-dk-unigram", "ocrqa-dyna_v1.1.0-da.bloom"))

PUNCT = "„•<>!\"#%&'’()*,./:;?[]\\~_{}\xa1\xab\xb7\xbb\xbf—^`¦§£=-"
TABLE = str.maketrans({c: " " for c in PUNCT} | {d: "0" for d in "0123456789"})

def ocrqa(text):
    words = set(unicodedata.normalize("NFKC", text).lower().translate(TABLE).split())
    return round(sum(w in known for w in words) / len(words), 3) if words else None

# %%
# test
test1 = ocrqa("Det var en mørk og stormfuld Nat, og Vinden hylede om Huset.")   # 1.0
test2 = ocrqa("Dct var en rnørk og ftormfuld Nat, og Vindcn hylede om Hnset.")  # 0.545
print(test1, test2)

# %%

# run it for each text 
df["ocrqa"] = df["text"].apply(ocrqa)

# %%

proxy_cols = ['wordcount_main_text_lex_auth', 
              'title_mention_lex_authorpage',
       'lex_author_mentioned_in_other_authorpage',
       'n_wikipedia_languages_auth', 
       'n_wikipedia_sitelinks_auth',
       'n_library_authorities_auth', 
       'n_wikidata_awards_auth',
       'n_wikidata_notable_works_auth', 
       'n_wikidata_works_auth',
       'title_mention_wikidata_notable_works', 
       'wordcount_wikipedia_da_auth',
       'wordcount_wikipedia_en_auth',
       'wikipedia_da_author_mentioned_in_other_authorpage',
       'wikipedia_en_author_mentioned_in_other_authorpage',

       'auth_mention_lithåndb_nameregistry',
       'auth_mention_littveje_nameregistry',
       'auth_mention_litt_find_nameregistry',
       'auth_mention_littmøn_nameregistry',
       'auth_mention_littgenveje_nameregistry',
       'auth_mention_bruglitt_nameregistry',
       'auth_mention_medtiden_nameregistry',
       'auth_mention_litt_intro_nameregistry',
       'auth_mention_trane_nameregistry', 
       'auth_mention_håndb_nameregistry',
       'auth_mention_rød_dan_lithis_nameregistry',
       'auth_mention_grøn_dan_lithis_nameregistry',
       'auth_mention_rød_dan_lithis_titleregistry',
       'title_mention_rød_dan_lithis_titleregistry', 
       'canon_culture_authors',
       'title_mention_canon_culture_authors', 
       'canon_educational_2025',
       'canon_educational_2004', 
       'author_mention_dsl_classic',
       'title_mention_canon_culture_authors_dsl_classic']

# correlation between proxy columns and ocrqa
# keep it one row
plt.figure(figsize=(15, 1))
corr = df[proxy_cols + ["ocrqa"]].corr().loc[["ocrqa"]]
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=False, annot_kws={"rotation": 90})
plt.savefig(FIGS / "ocrqa_correlation.png")
plt.show()
# %%

print(len(df))
# %%
