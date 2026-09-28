
I got the paper code and the blog post related to the work from ae studio and anthropic. 

The author talks about the duality of ai use how it can be used maliciously. The talk about all the consequences of restricting model access and the need for training separate models and how it is not possible when giving access to labs working on bio, nuclear, cyber capabilities. His argument goes like this If a bio lab wants to develop a drug they need top tier model capabilities which can also be used to make bio weapons. Now training and serving separate models would be the gold standard but it is prohibitively expensive. To address this issue the have built gradient routed auxiliary modules gram for short. lets see how they have designed these. 

a pretraining method that adds modules to a neural network and selectively updates them to induce specialization. 

![[Pasted image 20260805095701.png]]

According to the experiments conducted by the lab it shows that the model is much more resistant to changes and shown to be more robust than unlearning mechanisms and fine tuning mechanisms.  The say one of the major advantages of this approach is that this method scales very well. 

They talk about an alternative approach to deploying N separate models in N different deployment environments. The method is model branching: train a base model on filtered data then split into separate variants by fine tuning each on different dual use capability. 

![[Pasted image 20260805100927.png]]

Even with the above method we have serve them separately and lose out on most the capabilities that are learned during the pretraining at scale. 

So how do they intend to built the GRAM model?
Only in a single training run they intend to create multiple capability training profiles. By augmenting the MLP layers of a dense transformer with several smaller MLP modules. Then selectively enables forward and backward passes for different modules based on the data in the current training batch by directing capability specific updates to corresponding modules. During the inference they switch modules which selectively removes certain capabilities. According to their experiments that this method is vastly better than the post hoc unlearning in resisting malicious fine tuning. 


what did the GRAM use ?

-  Domain specific modules 
-  data dependent gradient selective updates 
- Seperate optimization process for parameter subsets 

what are approaches they are comparing this against ?

-  data filtering
-  model branching 
-  post hoc unlearning 


Background on how they have designed the engineering for the approach


They took $D = \{  D1, . . . , DN \}$ be the set of datasets and every set other than the first is the auxiliary capability dataset and D1 is the general purpose data set and they also took a capability profile. In this first one being the base or the retain set and the forget set being the $F= \{ 2,\dots,n \}$  as the auxiliaries to restrict. 

The validation cross entropy loss is defined to be $\ell(M,D_{i})$ 

![[Pasted image 20260805103225.png]]

How do we compare the amount of training compute required for the different modules specified to do this they have introduced a metric called as compute ratio metric. 

Definition: Let $M_{BL}$ denote a standard transformer block with dense MLP block trained on all the datasets $D_{all}=\cup_{i=1}^N{D_{i}}$. For each dataset they use a monotonic continuous power law learning curve $L_{s}$ that maps baseline training step s to baseline validation cross entropy loss at that step. 


$$ CR(M, D_{i}) = \frac{L_{i}^{-1}(\ell(M,D_{i})}{L_{i}^{-1}(\ell(M_{BL},D_{i})}
$$
a compute ration of r indicates that the model ac hives a loss equivalent to the baseline after a fraction r of the compute. 


How do we evaluate the results ?

for the forget sets lower lower compute rations indicate better performance and for higher compute ratios indicate for core and retain performance indicate better performance.


Now that the metrics are out of the way lets look at how we are going to design the architecture of the model to enable this modular approach !!!

So they took a Transformer architecture and augmented each MLP block with smaller auxiliary MLP modules. 

They are routing it based on data labels rather than per token or using a router 

![[Pasted image 20260805160821.png]]

The core module is almost the same size as the baseline transformer MLP Layer. Each of the auxiliary layer has the small fraction of the parameters of the base module. The intent is that the core module capture the general knowledge and the aux modules capture the specialized knowledge. Here we take the sum of the modules based on which of them are activated.

In a standard dense Transformer, each layer has an MLP $f_{\text{mlp}}: \mathbb{R}^d \to \mathbb{R}^d$. GRAM replaces this with a **gateless mixture**:

For layer $\ell$, define experts $E_0^{(\ell)}, E_1^{(\ell)}, \dots, E_{N-1}^{(\ell)}$ where:

- $E_0^{(\ell)}$ is the **core MLP** with hidden dimension $d_{\text{core}}$.
- $E_{i}^{(\ell)}$ for $i \geq 1$ is the auxiliary module for dataset $D_{i+1}$ with hidden dimension $d_{\text{aux}} \ll d_{\text{core}}$.

The layer output for input $x \in \mathbb{R}^{B \times T \times d}$ given forward mask $\mathbf{m}^{\text{fwd}} \in \{0,1\}^N$ is:

$$
\text{MoE}(x; \mathbf{m}^{\text{fwd}}) = \sum_{j=0}^{N-1} m^{\text{fwd}}_j \cdot E_j^{(\ell)}(x)
$$


So you might be wondering if this is the case how should the gradients be updated how exactly the training process happens? 

They used a method called as gradient routing for this. The forward pass depends on the dataset and the gradient updates over on the disjoint subsets as shown in the figure. 

They picked 3 different metrics to decide which of the modules be active how they should be trained. they are 
-  Auxiliary spread (probability with which the code module is updated in backward pass   pas)
- Auxiliary factor   this decide which of the aux module is selected paf
- core robustness (probability with which a random aux module gets activated pcr )


training the core batches 

when training the core batches they took the dataset D1 which is general purpose and trained it on the while activating one aux module with a probability of pcr in the forward pass. In the backward pass they just updated everything normally using the adamw optimizer. 
Sample a batch $b \sim D_1$. With probability $p_{\text{cr}}$, sample a random auxiliary index $k \sim \text{Uniform}\{2, \dots, N\}$.

**Forward mask:**
$$
m^{\text{fwd}}_j = 
\begin{cases}
1 & j = 0 \quad (\text{core always active})\\
1 & j = k \quad \text{with prob. } p_{\text{cr}} \\
0 & \text{otherwise}
\end{cases}
$$

**Backward mask:**
$$
m^{\text{bck}}_j = 
\begin{cases}
1 & j = 0 \quad (\text{core always updated})\\
1 & j = k \quad \text{if } m^{\text{fwd}}_k = 1 \\
0 & \text{otherwise}
\end{cases}
$$

All active parameters are updated. The random auxiliary activation forces the core to remain robust to the presence or absence of any single auxiliary.

while training on the auxiliary data 

when training on the auxiliary data they did a normal forward pass and while the backward pass they updated the core block with pas. The selected aux i is updated normally. 

Sample a batch $b \sim D_i$.

**Forward mask:**
$$
m^{\text{fwd}}_j = 
\begin{cases}
1 & j = 0 \quad (\text{core})\\
1 & j = i \quad (\text{auxiliary } i)\\
0 & \text{otherwise}
\end{cases}
$$

**Backward mask:**
$$
m^{\text{bck}}_j = 
\begin{cases}
1 & j = i \quad (\text{auxiliary } i \text{ always updated})\\
\mathbb{1}[\text{Bernoulli}(p_{\text{as}})] & j = 0 \quad (\text{core updated with prob. } p_{\text{as}})\\
0 & \text{otherwise}
\end{cases}
$$

When $p_{\text{as}} = 0$, the core is frozen and only $\Theta_{\text{aux}}^{(i)}$ learns $D_i$, maximizing modularity. When $p_{\text{as}} = 1$, the core absorbs auxiliary gradients, destroying modularity.

---



Running the experiments 
loading and extracting the data 
clone the repository

clone the repository

```python
!git clone https://github.com/agencyenterprise/modular-pretraining.git
```

**Output:**

```
Cloning into 'modular-pretraining'...
remote: Enumerating objects: 9590, done.
remote: Counting objects: 100% (4/4), done.
remote: Compressing objects: 100% (4/4), done.
remote: Total 9590 (delta 0), reused 0 (delta 0), pack-reused 9586 (from 1)
Receiving objects: 100% (9590/9590), 143.44 MiB | 15.67 MiB/s, done.
Resolving deltas: 100% (5325/5325), done.
Updating files: 100% (8058/8058), done.
```

get the data for the repository

I will be running only the stories section in the following notebook since the model is 24 million parameters and easy to understand but I will be doing a detailed breakdown of all the code presented in the experiment all architectures, methods used etc

to replicate the experiments please run the notebook in the kaggle notebooks with gpu p 100 enabled might need to check for torch compatability p100 use an old version of the torch but if everything goes well you will be able to train the model. One very important note to keep in mind is to enable presistence as the session will deactivate if unconnected you got a keep the tab open for the whole duration one simple trick is to keep the tab open at the start of the work day by the end of 10 hours you will be fine I suggest a 17min timer to check back on the progress and to keep the gpu engagaed. one of the other notes to keep in mind if you are running it on a free tier they will only allow for 30 hours per week so after the run make sure to shutdown the notebook once you are done

when the data is loaded download a copy to you local disk to avoid rerunning the data download process. save if you can.

code breakdown for the experiment This will take a while to understand by it is versy essential in your learning journey and one you will be able to replicate and run an architecture on your own the rest of it minor changes and scaffolding around the original code

I will be starting from each of the files and I will pick the execution order I have followed for running the experiment each of the function will be checked with dummy inputs and outputs so you will be able to understand

step 1 getting the data

```python
!python /content/modular-pretraining/src/data/prep_stories.py
```

**Output:**

```
out_dir: /content/modular-pretraining/src/data/stories
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
config.json: 100% 667/667 [00:00<00:00, 4.03MB/s]
tokenizer_config.json: 100% 591/591 [00:00<00:00, 3.32MB/s]
tokenizer.json: 100% 88.1k/88.1k [00:00<00:00, 90.6MB/s]
special_tokens_map.json: 100% 51.0/51.0 [00:00<00:00, 312kB/s]
README.md: 100% 4.18k/4.18k [00:00<00:00, 15.5MB/s]

data/train-00000-of-00007.parquet: downloading bytes:  74% 176M/238M [00:02<00:00, 148MB/s, 12.8MB/s  ]
data/train-00000-of-00007.parquet: downloading bytes:  91% 217M/238M [00:02<00:00, 191MB/s, 16.4MB/s  ]
data/train-00000-of-00007.parquet: reconstructing file:  56% 134M/238M [00:02<00:01, 63.0MB/s, 6.34MB/s  ]
data/train-00000-of-00007.parquet: downloading bytes: 100% 237M/237M [00:02<00:00, 85.7MB/s, 21.5MB/s  ]
data/train-00000-of-00007.parquet: reconstructing file: 100% 238M/238M [00:02<00:00, 86.2MB/s, 22.1MB/s  ]

data/train-00001-of-00007.parquet: downloading bytes:  86% 204M/238M [00:02<00:00, 154MB/s, 16.6MB/s  ]
data/train-00001-of-00007.parquet: reconstructing file:  56% 134M/238M [00:02<00:02, 50.1MB/s]
data/train-00001-of-00007.parquet: downloading bytes:  96% 228M/238M [00:02<00:00, 125MB/s, 20.4MB/s  ]
data/train-00001-of-00007.parquet: downloading bytes: 100% 237M/237M [00:04<00:00, 48.4MB/s, 20.4MB/s  ]
data/train-00001-of-00007.parquet: reconstructing file: 100% 238M/238M [00:04<00:00, 48.7MB/s, 18.8MB/s  ]

data/train-00002-of-00007.parquet: downloading bytes:  83% 197M/238M [00:02<00:00, 209MB/s, 13.7MB/s  ]
data/train-00002-of-00007.parquet: downloading bytes: 100% 236M/236M [00:02<00:00, 84.8MB/s, 21.8MB/s  ]
data/train-00002-of-00007.parquet: reconstructing file: 100% 238M/238M [00:02<00:00, 85.3MB/s, 22.2MB/s  ]

data/train-00003-of-00007.parquet: downloading bytes:  74% 176M/238M [00:02<00:00, 120MB/s, 15.4MB/s  ]
data/train-00003-of-00007.parquet: downloading bytes:  99% 236M/238M [00:05<00:00, 33.6MB/s, 16.5MB/s  ]
data/train-00003-of-00007.parquet: downloading bytes: 100% 236M/236M [00:06<00:00, 39.1MB/s, 16.7MB/s  ]
data/train-00003-of-00007.parquet: reconstructing file: 100% 238M/238M [00:06<00:00, 39.4MB/s, 18.4MB/s  ]

data/train-00004-of-00007.parquet: downloading bytes:  82% 194M/238M [00:02<00:00, 162MB/s, 13.6MB/s  ]
data/train-00004-of-00007.parquet: downloading bytes: 100% 237M/237M [00:03<00:00, 78.9MB/s, 21.2MB/s  ]
data/train-00004-of-00007.parquet: reconstructing file: 100% 238M/238M [00:03<00:00, 79.3MB/s, 22.1MB/s  ]

data/train-00005-of-00007.parquet: downloading bytes:  89% 212M/238M [00:03<00:00, 96.4MB/s, 18.0MB/s  ]
data/train-00005-of-00007.parquet: downloading bytes:  99% 236M/238M [00:03<00:00, 63.7MB/s, 19.5MB/s  ]
data/train-00005-of-00007.parquet: downloading bytes: 100% 236M/236M [00:04<00:00, 56.1MB/s, 19.5MB/s  ]
data/train-00005-of-00007.parquet: reconstructing file: 100% 238M/238M [00:04<00:00, 56.5MB/s, 21.2MB/s  ]

data/train-00006-of-00007.parquet: downloading bytes:  96% 229M/238M [00:04<00:00, 59.0MB/s, 18.5MB/s  ]
data/train-00006-of-00007.parquet: downloading bytes: 100% 236M/236M [00:04<00:00, 50.6MB/s, 18.8MB/s  ]
data/train-00006-of-00007.parquet: reconstructing file: 100% 238M/238M [00:04<00:00, 50.9MB/s, 21.3MB/s  ]

data/test-00000-of-00001.parquet: downloading bytes:  67% 11.3M/16.8M [00:01<00:00, 8.88MB/s,  406kB/s  ]
data/test-00000-of-00001.parquet: downloading bytes: 100% 16.7M/16.7M [00:01<00:00, 10.3MB/s, 1.61MB/s  ]
data/test-00000-of-00001.parquet: reconstructing file: 100% 16.8M/16.8M [00:01<00:00, 10.3MB/s, 1.62MB/s  ]
Generating train split: 100% 2115696/2115696 [00:24<00:00, 86987.92 examples/s] 
Generating test split: 100% 21371/21371 [00:00<00:00, 132684.30 examples/s]
Map (num_proc=20): 100% 1904126/1904126 [02:46<00:00, 11432.22 examples/s]
Map (num_proc=20): 100% 211570/211570 [00:20<00:00, 10282.28 examples/s]
Map (num_proc=20): 100% 2115696/2115696 [02:20<00:00, 15029.08 examples/s]
Dataset columns: ['topic', 'story', 'split']
Found 48 unique topic values
['a-deadline-or-time-limit', 'alien-encounters', 'bygone-eras', 'cultural-traditions', 'dinosaurs', 'dream-worlds', 'enchanted-forests', 'fairy-tales', 'fantasy-worlds', 'gardens', 'giant-creatures', 'haunted-places', 'hidden-treasures', 'holidays', 'invisibility', 'island-adventures', 'living-objects', 'lost-cities', 'lost-civilizations', 'magical-lands', 'magical-objects', 'miniature-worlds', 'mysterious-maps', 'mystical-creatures', 'outer-space', 'pirates', 'riddles', 'robots-and-technology', 'royal-kingdoms', 'school-life', 'seasonal-changes', 'secret-societies', 'shape-shifting', 'sibling-rivalry', 'snowy-adventures', 'space-exploration', 'sports', 'subterranean-worlds', 'superheroes', 'talking-animals', 'the-arts', 'the-sky', 'time-travel', 'treasure-hunts', 'undercover-missions', 'underwater-adventures', 'unusual-vehicles', 'virtual-worlds']
Map (num_proc=20): 100% 2115696/2115696 [35:43<00:00, 986.97 examples/s]
Map (num_proc=20): 100% 2115696/2115696 [03:46<00:00, 9360.38 examples/s]
Splitting by topic:   0% 0/48 [00:00<?, ?it/s]
Filter (num_proc=20):   0% 0/2115696 [00:00<?, ? examples/s]
Filter (num_proc=20):   0% 1000/2115696 [00:04<2:40:15, 219.91 examples/s]
Filter (num_proc=20):   0% 7000/2115696 [00:04<17:14, 2038.11 examples/s] 
Filter (num_proc=20):   1% 12000/2115696 [00:04<08:41, 4032.87 examples/s]
Filter (num_proc=20):   1% 20000/2115696 [00:04<04:13, 8262.10 examples/s]
Filter (num_proc=20):   1% 26000/2115696 [00:07<07:16, 4783.79 examples/s]
Filter (num_proc=20):   1% 30000/2115696 [00:07<05:40, 6122.72 examples/s]
Filter (num_proc=20):   2% 36000/2115696 [00:07<03:53, 8896.16 examples/s]
Filter (num_proc=20):   2% 40000/2115696 [00:07<03:12, 10765.41 examples/s]
`````



```
understanding the script

the format of the json for the stories metadata

the file is present in the folder - src/data/stories/metdata.json

```python
{
  "<category_name>": {
    "train": {
      "total_tokens": <integer>,
      "example": "<story text ending with [EOS]>"
    },
    "test": {
      "total_tokens": <integer>,
      "example": "<story text ending with [EOS]>"
    }
  },
  ...
}
```

the category name is the name of the story type

moving onto the actual data download and processing stage of the pipeline

the execution code is the file prep_stories.py in the folder src.data.stories

the libraries we will be using here are

- argparse - for parsing the arguments
- JSON - FOR JSON PROCESSING
- OS - FOR directory and file handling
- shutil - for directory handling
- pathlib - for fixing setting path variables
- typing - for type safety and from typing we use Any Dict List and Literal (for exactly matching the arguments )
- collections - for ordrdered dictionary processing
- numpy for numerical python faster processing
- datasets - for downloading and manupulating large datasets
- dotenv - for environment file loading
- hugginface_hub - for downloading the actual dataset
- tqdm - tqdm for interactive processing bar
- transformers - for the tokenizer we use autotokenizer here
- transformer.utils - for looging the logs

```python
#!/usr/bin/env python
"""
Prepare token-level binary shards and write a metadata.json file.
"""
import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Literal
from collections import OrderedDict

import numpy as np
from datasets import Dataset, concatenate_datasets, load_dataset
from dotenv import load_dotenv
from huggingface_hub import HfApi, hf_hub_download
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from transformers.utils import logging
```

setting the path to the environment varaiables and the verbosity of logging

```python

logging.set_verbosity(40)

# Load environment variables
load_dotenv(Path(__file__).parent.parent.parent / ".env")
```

breaking down function by function

we wanna write the files to bin files

takes 3 input arguments and produces none

takes the path of the file array and the data type and writes it to a memeory mapped file

- np.memmap creates an array that exists on disk, but can be accessed like a regular NumPy array in memory.
    
- mode="w+" means: create the file if it doesn’t exist, overwrite it if it does, and allow both reading and writing.
    
- shape=(total,) – a one‑dimensional array with exactly total elements.
    
- The file will contain raw binary data (no headers, no separators), just the numbers stored consecutively.
    
- tqdm shows a progress bar as the loop iterates over the list of sub‑lists. It’s not essential for the logic, but helpful for large datasets.
    
- Each sub‑list a is assigned to the appropriate slice of the memory‑mapped array. This copies the integers into the file at the correct positions.
    
- The index idx keeps track of where the next sub‑list should start, advancing by the length of the current sub‑list.
    

advantage of doing this way saves to disk and helps not lpoad the entire thing into ram and faster writing speed

```python
def memmap_write(
    fname: Path,
    arr: List[List[int]],
    dtype: np.dtype = np.uint16,
) -> None:
    """
    Write array data to a memory-mapped file.

    Args:
        fname: Path to output file
        arr: List of arrays to write
        dtype: NumPy data type for the memory-mapped array
    """

    # getting the total length
    total = sum(len(a) for a in arr)
    # writing to the disk
    mmap = np.memmap(fname, dtype=dtype, mode="w+", shape=(total,))
    idx = 0
    for a in tqdm(arr, desc="writing", total=len(arr)):
        mmap[idx : idx + len(a)] = a
        idx += len(a)
    mmap.flush()
```

Preparing the data to be the right format

prep function takes inputs

- num_proc - number of process that will be running
- tokenizer - the tokenizer in this case autotokenizer
- max_length - maximum allowable length of the sequence
- length_strategy - maximum context length which can be truncated or dropped or simply none
- sample pct - default 1 and pct 1 full pct 0 none and the values range from 0 to 1 and contain how much we wanna use
- column - name of the column (in this case category of the story)

names of the varaiables

#### section 1

- dset_name - name of the dataset
- ds - loading the dataset onto the dataframe with load_dataset
- if they requested as seen above in the json format for a sample of the dataset we will print that percentage of the sample using the dataframe.select method
- from the dataframe we will only keep the column story - "string of stories " and column which is the name of the story type (column from the input)
- once the appropriate columns are selected for the dataframe we will do a simple train test split and do a slight modification of the column text for naming the files properly suppose Fairy TALES will be converted to fairy-tales for easier file naming

#### section 2

- define a labels varaibles and get the unique columns from the column (story categories)
    
- tok_fn the train test contents in dictionary and encode them using the tokenizer
    
- append end of the text token at the end
    
- from the above input category if truncate for length strategy then truncate this the specified max length and append the eos of token
    
- if dropping is enabled them drop stories that exceed the max length and if non just simply process it
    

#### section 3

- now we take data a ordered dictionary and map each of the colum in the labels with a unique identifer integer starting from 0 and using this value match all the rows into the specified category and filter through the dataframe

and onnce that is done we add simply label and train test files like the json format presented above

```python


def prep(
    num_proc: int,
    tokenizer: AutoTokenizer,
    max_length: int,
    column: str,
    length_strategy: Literal["truncate", "drop", "none"],
    sample_pct: float = 1.0,
) -> Dict[str, Dataset]:

#-------------------------------------------------------section 1------------------------------------------------------------------------

    dset_name = "SimpleStories/SimpleStories"
    ds = load_dataset(dset_name, split="train")

    # Sample dataset if requested
    if sample_pct < 1.0:
        sample_size = int(len(ds) * sample_pct)
        ds = ds.select(range(sample_size))
        print(f"Sampling {sample_pct*100}% of data: {sample_size} examples")

    #keep only the column "column" plus "story"
    ds = ds.select_columns([column, "story"])

    splits = ds.train_test_split(test_size=0.1, seed=42) #NOTE test size was 0.01 in OG version
    train, test = splits["train"], splits["test"]

    train = train.map(lambda ex: {"split": "train"}, num_proc=num_proc)
    test = test.map(lambda ex: {"split": "test"}, num_proc=num_proc)

    ds = concatenate_datasets([train, test])

    # For the col "column", replace all values in that col with the value in that col replaced with dashes
    ds = ds.map(lambda ex: {column: str(ex[column]).lower().replace(' ', '-')}, num_proc=num_proc)

    print("Dataset columns:", ds.column_names)

#------------------------------------------------------------section 2 -------------------------------------------------------------------





    # --------------------------------------------------------- #
    # 1. tokenisation                                           #
    # --------------------------------------------------------- #

    # Build label list
    labels = sorted(ds.unique(column))
    print(f"Found {len(labels)} unique {column} values")
    print(labels)

    def tok_fn(ex: Dict[str, Any]) -> Dict[str, Any]:

        ids = tokenizer.encode(ex["story"], add_special_tokens=False)
        ids.append(tokenizer.eos_token_id)

        if length_strategy == "truncate" and max_length > 0:
            ids = ids[:max_length]
            ids[-1] = tokenizer.eos_token_id

        return {"ids": ids, "len": len(ids)}

    ds = ds.map(tok_fn, num_proc=num_proc)

    # If dropping is enabled, remove stories longer than max_length
    if length_strategy == "drop" and max_length > 0:
        ds = ds.filter(lambda ex: ex["len"] <= max_length, num_proc=num_proc)



#--------------------------------------------------section 3 ------------------------------------------------------------------------------------



    # --------------------------------------------------------- #
    # 2. value mapping                                          #
    # --------------------------------------------------------- #

    # value -> value_id (makes subsetting much faster)
    ds = ds.map(lambda ex: {f"{column}_id": labels.index(ex[column])}, num_proc=num_proc)

    data = OrderedDict()
    for label in tqdm(labels, desc=f"Splitting by {column}"):
        value_id = labels.index(label)
        subset = ds.filter(lambda ex, value_id=value_id: ex[f"{column}_id"] == value_id, num_proc=num_proc)
        train = subset.filter(lambda ex: ex["split"] == "train", num_proc=num_proc)
        test = subset.filter(lambda ex: ex["split"] == "test", num_proc=num_proc)
        data[label] = {
            "train": train,
            "test": test,
        }

    return data
```

Moving onto the write function

inputs for the function

- datasets - the processed dataframe from prep
- column - name of the column
- out_dir - path to the directory where the files will be written to the bin files
- tokenizer_name - name of the tokenizer for the config files later
- length_strategy - defined in the above truncate drop and none

#### section 1

- define a varaible meta for the meta data about a category
- initalize total_tokens_train and total_tokens_test
- get the labels remove file the already exists in the directory memmap_write
- iterate over the loop and count the token in each of the file category

#### section 2

writing the meta data and writing the json file

- define meta data file with
- total_tokens_train for the count of total tokens for the train category
- total_tokens_test for the count of total tokens for the test category
- tokenizer - name of the tokenizer that we give as defined above
- max_length - defined above
- column - defined above
- labels - defined above
- length strategy - defined above once this is done

use the path to +w write the files to the metadata.json - with the above format json for checking if all the categories saved properly

```python


def write(
    datasets: Dict[str, Dataset],
    column: str,
    out_dir: Path,
    max_length: int,
    tokenizer_name: str,
    tokenizer: AutoTokenizer,
    length_strategy: Literal["truncate", "drop", "none"],
) -> None:
    """Write datasets to binary files and collect metadata."""



#----------------------------------------------------------section 1 -------------------------------------------------------------------------
    meta: dict[str, Any] = {}
    total_tokens_train = 0
    total_tokens_test = 0
    labels = sorted(list(datasets.keys()))

    for label, splits in datasets.items():

        print("label", label)
        print("splits", splits)

        meta[label] = {
            "train": {},
            "test": {},
        }

        for split in ["train", "test"]:

            subset = splits[split]
            out_path = out_dir / f"{label}_{split}.bin"
            if out_path.exists():
                os.remove(out_path)

            # write tokens
            memmap_write(
                out_path,
                subset["ids"],
                np.uint16,
            )

            # ---------- per‑split statistics ----------
            total_tokens = int(np.sum(subset["len"]))
            example_text = tokenizer.decode(subset[-1]["ids"], skip_special_tokens=False)

            meta[label][split] = {
                "total_tokens": total_tokens,
                "example": example_text,
            }

            if split == "train":
                total_tokens_train += total_tokens
            else:
                total_tokens_test += total_tokens


#---------------------------------------------------- section 2 ---------------------------------------------------------------------------------

    # ---------- global statistics ----------
    meta["all"] = {
        "total_tokens_train": total_tokens_train,
        "total_tokens_test": total_tokens_test,
        "tokenizer": tokenizer_name,
        "vocab_size": len(tokenizer),
        "max_length": max_length,
        "column": column,
        "labels": labels,
        "length_strategy": length_strategy,
    }

    # ---------------------------------------------------- #
    # dump metadata.json                                   #
    # ---------------------------------------------------- #
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False, default=str)
```

Moving onto the run fuction

this is the main function where the code actually starts running once we have done the detailed breakdown of the data processing pipeline it is almost the same for all the data being loaded with very few minor changes in the experiment they used fine web, stories, and papers script here papers dataset is very intresting it contains data for nuclear, cyber, bioweapons categories but to train you might need to fight 800 M parameter model trying running it if you can aat the end of the notebook I will add little details on how to run the experiment as well

inputs for the following function

- out_dir - directory where the output files will be saved
- num_proc - number of processes to run
- column - defined above
- max_length - defined above
- length strategy - defined above
- tokenizer name - defined above
- download_bins- this parameter lets you download the bins its a boolean flag
- upload_bins - this parameter lets you upload the bin files its a boolen flag
- sample_pct - default full but to display we show certain percentage of the row

```python

# --------------------------------------------------------------------------- #
# main preparation sequence                                                   #
# --------------------------------------------------------------------------- #


def run(
        out_dir: Path | None,
        num_proc: int,
        column: str,
        max_length: int,
        length_strategy: str,
        tokenizer_name: str,
        download_bins: bool,
        upload_bins: bool,
        sample_pct: float = 1.0,
    ) -> None:

    default_out_dir = Path(__file__).parent / "stories"
    if out_dir is None:
        out_dir = default_out_dir

    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if out_dir.resolve() != default_out_dir.resolve():

        if default_out_dir.is_symlink() or default_out_dir.exists():
            default_out_dir.unlink() if default_out_dir.is_symlink() else shutil.rmtree(default_out_dir)
        default_out_dir.parent.mkdir(parents=True, exist_ok=True)
        default_out_dir.symlink_to(out_dir.resolve())
        print(f"Symlinked {default_out_dir} -> {out_dir.resolve()}")

    print("out_dir:", out_dir)

    # Get HF token from environment
    hf_token = os.getenv("HF_TOKEN", None)
    repo_id = "erol-AE/GR-MoE"
    subfolder = "stories"

    # Download bins if requested
    if download_bins:
        print(f"Downloading .bin files from {repo_id}/{subfolder}...")
        api = HfApi(token=hf_token)

        # List all files in the subfolder
        try:
            repo_files = api.list_repo_files(repo_id=repo_id, repo_type="dataset", token=hf_token)
            bin_files = [f for f in repo_files if f.startswith(subfolder) and f.endswith('.bin')]

            # Also download metadata.json
            metadata_files = [f for f in repo_files if f.startswith(subfolder) and f.endswith('metadata.json')]

            all_files = bin_files + metadata_files

            if not all_files:
                print(f"No .bin or metadata.json files found in {repo_id}/{subfolder}")
            else:
                for file_path in tqdm(all_files, desc="Downloading files"):
                    hf_hub_download(
                        repo_id=repo_id,
                        filename=file_path,
                        repo_type="dataset",
                        token=hf_token,
                        local_dir=out_dir.parent,
                        local_dir_use_symlinks=False,
                    )
                    print(f"Downloaded {file_path} to {out_dir / Path(file_path).name}")

                print("Download complete!")
        except Exception as e:
            print(f"Error downloading files: {e}")
            raise

        return

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    data = prep(
        num_proc=num_proc,
        tokenizer=tokenizer,
        column=column,
        max_length=max_length,
        length_strategy=length_strategy,
        sample_pct=sample_pct,
    )

    # Write datasets and metadata
    write(
        datasets=data,
        column=column,
        out_dir=out_dir,
        max_length=max_length,
        tokenizer_name=tokenizer_name,
        tokenizer=tokenizer,
        length_strategy=length_strategy,
    )

    print("Done - binary shards + metadata.json written to", out_dir)

    # Upload bins if requested
    if upload_bins:
        print(f"Uploading .bin files to {repo_id}/{subfolder}...")
        api = HfApi(token=hf_token)

        # Ensure repo exists (will not error if it already exists)
        try:
            api.create_repo(repo_id=repo_id, token=hf_token, exist_ok=True, repo_type="dataset")
            print(f"Dataset repository {repo_id} ready")
        except Exception as e:
            print(f"Note: Could not create/verify repo (it may already exist): {e}")

        # Find all .bin files and metadata.json in out_dir
        bin_files = list(out_dir.glob("*.bin"))
        metadata_file = out_dir / "metadata.json"

        files_to_upload = bin_files.copy()
        if metadata_file.exists():
            files_to_upload.append(metadata_file)

        if not files_to_upload:
            print(f"No .bin or metadata.json files found in {out_dir} to upload")
        else:
            for file_path in tqdm(files_to_upload, desc="Uploading files"):
                try:
                    api.upload_file(
                        path_or_fileobj=str(file_path),
                        path_in_repo=f"{subfolder}/{file_path.name}",
                        repo_id=repo_id,
                        repo_type="dataset",
                        token=hf_token,
                    )
                    print(f"Uploaded {file_path.name} to {repo_id}/{subfolder}")
                except Exception as e:
                    print(f"Error uploading {file_path.name}: {e}")
                    raise

            print("Upload complete!")
```

#### cli arguments

we use the argparse Argument parser (

- --out_dir - directory to write the bin files to
- --num_proc - the number of processes that needs to run
- --column - name of the column
- --max-length - length of max sequence
- --length_strategy - truncate, drop, none
- --tokenizer - tokenizer for the data
- -- download_bins - download the bin files
- -- upload_bins - upload the bin files
- -- sample - using the proc percentage print ceratin sample text

)

```python


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":

    ap = argparse.ArgumentParser("Prepare simple stories")
    ap.add_argument("--out_dir", default=None, help="directory to write .bin files")
    ap.add_argument("--num_proc", type=int, default=20)
    ap.add_argument("--column", type=str, default="topic")
    ap.add_argument("--max_length", type=int, default=-1)
    ap.add_argument("--length_strategy", type=str, default="none", choices=["truncate", "drop", "none"])
    ap.add_argument("--tokenizer", type=str, default="SimpleStories/SimpleStories-1.25M")
    ap.add_argument("--download_bins", action="store_true")
    ap.add_argument("--upload_bins", action="store_true")
    ap.add_argument("--sample", type=float, default=1.0, help="Fraction of data to use (0.0-1.0), e.g., 0.01 for 1%%")
    args = ap.parse_args()

    run(
        out_dir=args.out_dir,
        num_proc=args.num_proc,
        column=args.column,
        max_length=args.max_length,
        length_strategy=args.length_strategy,
        tokenizer_name=args.tokenizer,
        download_bins=args.download_bins,
        upload_bins=args.upload_bins,
        sample_pct=args.sample,
    )
```





































the code blocks 
```python 
def freeze(module: nn.Module, do_freeze: bool):
    """Return `module` unchanged if do_freeze=False, else a callable that runs
    the module via functional_call with all params/buffers detached.
    Activations still flow through (so downstream experts can train), but
    the module's own parameters don't accumulate .grad."""
    if not do_freeze:
        return module
    params_and_bufs = {
        **{n: p.detach() for n, p in module.named_parameters()},
        **{n: b for n, b in module.named_buffers()},
    }
    def call(*args, **kwargs):
        return torch.func.functional_call(module, params_and_bufs, args, kwargs)
    return call
```


here when do_freeze is true then the parameters are detached and the gradients does not accumulate on this 

```python 
class MoE(nn.Module):
    """
    Gateless MoE. Dispatch by per-sample multi-hot mask over experts.
    """
    def __init__(
        self,
        embed_dim: int,
        num_experts: int,
        core_dim: int,
        aux_dim: int,
    ) -> None:
        super().__init__()

        self.experts = nn.ModuleList(
            [MLP(embed_dim, core_dim)] +        # core (idx 0)
            [MLP(embed_dim, aux_dim)           # aux  (idx 1..E-1)
             for _ in range(num_experts - 1)]
        )

    def forward(
        self, x: torch.Tensor, 
        fwd_mask: torch.Tensor,      # (K,) — which experts are ACTIVE in forward
        bck_mask: torch.Tensor,      # (K,) — which experts get GRADIENTS in backward
    ) -> torch.Tensor:
        """
        x:        (B, T, E)
        fwd_mask: per-expert forward weights. Boolean multi-hot during training.
        bck_mask: boolean, multi-hot selection of experts for the backward pass.
        returns:  (B, T, E)
        """

        # Fast path: all experts active (training baseline / all-active case)
        if bool((fwd_mask == 1).all()) and bool((bck_mask == 1).all()):
            K = len(self.experts)

            # Core output
            y = self.experts[0](x)

            # Batched forward for all aux experts at once (efficiency)
            aux_fc_w = torch.stack([e.c_fc.weight for e in self.experts[1:]], dim=0)
            aux_fc_b = torch.stack([e.c_fc.bias for e in self.experts[1:]], dim=0)
            aux_proj_w = torch.stack([e.c_proj.weight for e in self.experts[1:]], dim=0)
            aux_proj_b = torch.stack([e.c_proj.bias for e in self.experts[1:]], dim=0)
            
            aux_hidden = torch.einsum('bte,khe->btkh', x, aux_fc_w) + aux_fc_b.view(1, 1, K-1, -1)
            aux_hidden = F.gelu(aux_hidden, approximate="tanh")
            aux_output = torch.einsum('btkh,keh->bte', aux_hidden, aux_proj_w) + aux_proj_b.sum(dim=0)
            
            return y + aux_output

        else:
            # Slow path: selective activation / ablation
            y = None
            for i in range(len(self.experts)):
                w = fwd_mask[i]
                if bool(w):  # nonzero forward weight
                    # FREEZE if this expert is NOT in bck_mask — forward runs, but no grad
                    expert = freeze(self.experts[i], do_freeze=not bool(bck_mask[i]))
                    out = expert(x)
                    if not bool(w == 1):  # scale for fractional titration weights
                        out = w * out
                    y = out if y is None else y + out
            return y
```

| Component               | File                                                | What it does                                                                                                                       |
| ----------------------- | --------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| **Architecture**        | `src/model/moe.py`                                  | Replaces `MLP` with `MoE`: core MLP + aux MLPs. `freeze()` blocks gradients via `torch.func.functional_call` with detached params. |
| **Parameter partition** | `src/model/moe.py`                                  | `get_params(label)` yields disjoint subsets: core = embeddings + attention + norms + core MLP; aux\_i = only aux\_i MLPs.          |
| **Training loop**       | `src/run/train/routed.py`                           | Builds `fwd_mask` / `bck_mask` per batch. Calls `model.forward(tokens, targets, fwd_mask, bck_mask)`.                              |
| **Separate optimizers** | `src/run/train/routed.py`                           | One `AdamW` per label. Only step optimizers whose params have non-None grads.                                                      |
| **Batch policy**        | Experiment configs (`stories/methods/run.py`, etc.) | Defines `p_as` (aux spread), `p_cr` (core robustness), and sampling frequencies.                                                   |

