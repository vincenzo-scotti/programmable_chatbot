import random

import bz2
import pickle

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, AutoTokenizer

from programmable_chatbot.utils import IGNORE_INDEX
from .utils import _DialogueCorpus
from .corpora import *

from typing import List, Dict, Union


# Corpus mapping
CORPORA: Dict = {
    DailyDialog.IDENTIFIER: DailyDialog,
    EmpatheticDialogues.IDENTIFIER: EmpatheticDialogues,
    PersonaChat.IDENTIFIER: PersonaChat,
    WizardOfWikipedia.IDENTIFIER: WizardOfWikipedia,
    IEMOCAP.IDENTIFIER: IEMOCAP,
    TopicalChat.IDENTIFIER: TopicalChat,
    CounsellingAndPsychotherapyCorpus.IDENTIFIER: CounsellingAndPsychotherapyCorpus,
    HOPE.IDENTIFIER: HOPE,
    EPITOME.IDENTIFIER: EPITOME,
    CounselChat.IDENTIFIER: CounselChat
}


class PromptedOpenDomainDialogues(Dataset):

    def __init__(
            self,
            corpora_dir_path: str,
            tokenizer: Union[str, PreTrainedTokenizer],
            split: Literal['train', 'validation', 'test'],
            cache_dir_path: str,
            corpus_prefix: str = 'podd',
            corpus_list: Optional[List[str]] = None,
            evaluation: bool = False,
            device: torch.device = torch.device('cpu'),
            joblib_backend: str = 'threading',
            n_jobs: int = -1,
            verbosity_level: int = 0,
            corpus_kwargs: Optional[Dict[str, Dict]] = None,
            **kwargs
    ):
        super(PromptedOpenDomainDialogues, self).__init__()
        # Data set
        # Split identifier
        self.split: Literal['train', 'validation', 'test'] = split
        # Path to cache
        self.corpus_cache_file_path: str = os.path.join(cache_dir_path, f'{corpus_prefix}_{split}.pbz2')

        self.evaluation: bool = evaluation

        # Collating and data preparation
        # Tokeniser to prepare inputs
        if isinstance(tokenizer, str):
            self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(tokenizer)
        else:
            self.tokenizer: PreTrainedTokenizer = tokenizer
        # Additional info
        self.device: torch.device = device

        # Data
        self.data: Union[List[str], ]

        # Generate cache if needed
        if not os.path.exists(self.corpus_cache_file_path):
            # Save parallelisation options
            self.joblib_backend: str = joblib_backend
            self.n_jobs: int = n_jobs
            self.verbosity_level: int = verbosity_level
            # Save current prefix
            self.prefix = corpus_prefix
            # Create cache dir if not exists
            if not os.path.exists(cache_dir_path):
                os.mkdir(cache_dir_path)
            # Save path to raw data
            self.corpora_dir_path: str = corpora_dir_path
            # Get corpus list ad list of all available corpora if not provided
            if corpus_list is None:
                self.corpus_list: List[str] = [
                    dir_name for dir_name in os.listdir(corpora_dir_path)
                    if os.path.isdir(os.path.join(corpora_dir_path, dir_name))
                ]
            # Else simply save the provided list
            else:
                self.corpus_list: List[str] = corpus_list
            # Load all corpora and generate cache
            self._generate_data_cache(corpus_kwargs if corpus_kwargs is not None else dict(), **kwargs)
        # Otherwise simply load the cache
        else:
            self._load_data_cache()

    def __len__(self) -> int:
        if self.evaluation:
            raise ValueError('Corpus loaded in evaluation mode')
        # Number of sequences within the data set
        return len(self.data)

    def __getitem__(self, index: int) -> str:
        if self.evaluation:
            raise ValueError('Corpus loaded in evaluation mode')
        # Get utterances from data set
        return self.data[index]

    def _generate_data_cache(self, corpus_kwargs, **kwargs):
        # Create corpora instances
        corpora: List[_DialogueCorpus] = [
            CORPORA[corpus_id](
                os.path.join(self.corpora_dir_path, corpus_id),
                self.split,
                self.tokenizer,
                joblib_backend=self.joblib_backend,
                n_jobs=self.n_jobs,
                verbosity_level=self.verbosity_level,
                **kwargs,
                **corpus_kwargs.get(corpus_id, dict())
            )
            for corpus_id in self.corpus_list
        ]
        if self.evaluation:
            self.data = {
                corpus.IDENTIFIER: corpus.get_data_for_evaluation() for corpus in corpora
            }
        else:
            # Gather samples
            with parallel_backend(self.joblib_backend, n_jobs=self.n_jobs):
                data = sum(Parallel(verbose=self.verbosity_level)(
                    delayed(corpus.get_data_for_fitting)() for corpus in corpora
                ), list())

            # Merge together short sequences
            self.data = self._postprocess(data)

        # Save compressed pickle file with data set
        with bz2.BZ2File(self.corpus_cache_file_path, 'w') as f:
            pickle.dump(self.data, f)

    def _postprocess(self, data: List[str]):
        # Compute the length in tokens of all sequences
        with parallel_backend(self.joblib_backend, n_jobs=self.n_jobs):
            tmp_data = sorted(Parallel(verbose=self.verbosity_level)(
                delayed(lambda x: (x, len(self.tokenizer(sample).input_ids)))(sample) for sample in data
            ), key=lambda x: -x[1])
        # Iteratively merge sequences until no update is possible # TODO find better solution
        data = []
        while len(tmp_data) > 0:
            # If there are at least two samples check the lengths
            if len(tmp_data) > 1:
                # It the length of the longest exceeds the maximum remove it
                if tmp_data[0][1] + tmp_data[-1][1] >= self.tokenizer.model_max_length:
                    data.append(tmp_data.pop(0)[0])
                # Else merge the first and last
                else:
                    s, l = tmp_data.pop(-1)
                    if random.choice([True, False]):
                        if tmp_data[0][0].endswith(self.tokenizer.eos_token):
                            conj = ('', 0)
                        else:
                            conj = (self.tokenizer.eos_token, 1)
                        tmp_data[0] = (tmp_data[0][0] + conj[0] + s, tmp_data[0][1] + conj[1] + l)
                    else:
                        if s.endswith(self.tokenizer.eos_token):
                            conj = ('', 0)
                        else:
                            conj = (self.tokenizer.eos_token, 1)
                        tmp_data[0] = (s + conj[0] + tmp_data[0][0], l + conj[1] + tmp_data[0][1])
            # Else if only one sample is remaining remove it
            else:
                data.append(tmp_data.pop(0)[0])

        return data

    def _load_data_cache(self):
        # Load compressed pickle file
        with bz2.BZ2File(self.corpus_cache_file_path, 'r') as f:
            self.data = pickle.load(f)

    def collate(self, utterances: List[str]):
        # Prepare input tokens
        input_encodings = self.tokenizer(utterances, return_tensors='pt', padding=True, truncation=True)
        # Prepare outputs
        labels = torch.clone(input_encodings.input_ids)
        labels[~input_encodings.attention_mask.bool()] = IGNORE_INDEX

        return input_encodings, labels
