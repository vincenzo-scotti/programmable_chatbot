import os
import pickle

import re
import random
from itertools import cycle, repeat

from joblib import Parallel
from joblib import delayed
from joblib import parallel_backend

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from typing import List, Dict, Union, Optional, Literal, Tuple

IGNORE_INDEX = -1

# Placeholders
PLACEHOLDER_F_STRING: str = '<|{}|>'
INTERACTION: str = PLACEHOLDER_F_STRING.format('interaction')
SPEAKERS: str = PLACEHOLDER_F_STRING.format('speakers')
SPEAKER_DESCRIPTIONS: str = PLACEHOLDER_F_STRING.format('speakerswdescription')
CONTEXT: str = PLACEHOLDER_F_STRING.format('context')
RESPONSE: str = PLACEHOLDER_F_STRING.format('response')
DIALOGUE_CHUNK: str = PLACEHOLDER_F_STRING.format('chunk')
LABEL_ID: str = PLACEHOLDER_F_STRING.format('labelid')
LABEL_DESCRIPTION: str = PLACEHOLDER_F_STRING.format('labeldescription')
LABEL_VALUES: str = PLACEHOLDER_F_STRING.format('labelvalues')
LABEL_VALUE: str = PLACEHOLDER_F_STRING.format('labelvalue')
CLS_SEP: str = PLACEHOLDER_F_STRING.format('sepdiscriminator')


def capitalize(s: str) -> str:
    return s[:1].upper() + s[1:]


# Abstract corpus class
class _DialogueCorpus(Dataset):
    # Corpus identifier placeholder
    IDENTIFIER: str = None
    # Substitutes for non-unicode special characters
    UNICODE_SWITCH_LIST: List[Tuple[bytes, str]] = [
        ("\u2019", "'"),
        ("\u2018", "'"),
        ("\u201d", '"'),
        ("\u201c", '"'),
        ("\u2014", "--"),
        ("\u2013", "--"),
        ("\u3002", ". "),
        ("\u2032", "'"),
        ("\u3001", ", ")
    ]
    # Composable strings to generate the dialogue
    INTERACTION_TYPE: List[str] = ['conversation', 'dialogue', 'chit-chat']
    CHUNK_TYPE: List[str] = ['window', 'chunk', 'piece', 'passage']
    CONTEXT_TYPE: List[str] = ['context', 'history', 'past']
    RESPONSE_TYPE: List[str] = ['response', 'utterance']
    # Premise
    PREMISE: List[str] = [
        f'The following is a {INTERACTION} between {SPEAKER_DESCRIPTIONS}.',
        f'Here follows a {INTERACTION} between {SPEAKER_DESCRIPTIONS}.',
        f'This is a {INTERACTION} between {SPEAKER_DESCRIPTIONS}.',
        f'The following is the transcript of a {INTERACTION} between {SPEAKER_DESCRIPTIONS}.'
    ]
    HUMAN_PARTICIPANTS: List[str] = ['two speakers', 'two humans', 'two human speakers', 'two persons', 'two users']
    # 'speaker': ['speakers', 'humans'], 'person': ['persons', 'humans'], 'user': ['users', 'humans']
    HUMAN_PARTICIPANT: List[str] = ['a human speaker', 'a human', 'a person', 'a human user']
    # 'speaker': ['a human speaker', 'a human'], 'person': ['a person', 'a human'], 'user': ['a human user', 'a human']
    BOT_PARTICIPANT: List[str] = [
        'an intelligent AI assistant', 'an AI assistant', 'an intelligent AI system', 'an AI system', 'an AI',
        'an intelligent chatbot', 'a chatbot', 'an intelligent conversational agent', 'a conversational agent',
        'an intelligent dialogue agent', 'a dialogue agent'
    ]
    # 'AI': ['an intelligent AI assistant', 'an AI assistant', 'an intelligent AI system', 'an AI system', 'an AI'],
    # 'chatbot': ['an intelligent chatbot', 'a chatbot', 'an intelligent conversational agent', 'a conversational agent', 'an intelligent dialogue agent', 'a dialogue agent']
    PRESENTATION: List[str] = ['called', 'namely', 'referred to as', 'hereafter referred to as']
    # Task description
    GENERATIVE_TASK_PREFIX: List[str] = [
        f'In the following interactions, {SPEAKERS} will converse in natural language.',
        f'During these interactions, {SPEAKERS} will converse in natural language.',
        f'In the following interactions, {SPEAKERS} converse in natural language.',
        f'During these interactions, {SPEAKERS} converse in natural language.',
        f'In the following interactions, {SPEAKERS} are conversing in natural language.',
        f'During these interactions, {SPEAKERS} are conversing in natural language.'
    ]
    DISCRIMINATIVE_TASK_PREFIX: Dict[str, List[str]] = {
        'global': [
            f'Label the following {INTERACTION} {DIALOGUE_CHUNK} with information about',
            f'Label the following {INTERACTION} {DIALOGUE_CHUNK} with info about',
            f'Label the following {INTERACTION} {DIALOGUE_CHUNK} with',
            f'Classify the following {INTERACTION} {DIALOGUE_CHUNK} with information about',
            f'Classify the following {INTERACTION} {DIALOGUE_CHUNK} with info about',
            f'Classify the following {INTERACTION} {DIALOGUE_CHUNK} with',
            f'Annotate the following {INTERACTION} {DIALOGUE_CHUNK} with information about',
            f'Annotate the following {INTERACTION} {DIALOGUE_CHUNK} with info about',
            f'Annotate the following {INTERACTION} {DIALOGUE_CHUNK} with'
        ],
        'local': [
            f'Label the following {RESPONSE} of a {INTERACTION} with information about',
            f'Label the following {RESPONSE} extracted from a {INTERACTION} with information about',
            f'Label the following {RESPONSE} of a {INTERACTION} with info about',
            f'Label the following {RESPONSE} extracted from a {INTERACTION} with info about',
            f'Label the following {RESPONSE} of a {INTERACTION} with',
            f'Label the following {RESPONSE} extracted from a {INTERACTION} with',
            f'Classify the following {RESPONSE} of a {INTERACTION} with information about',
            f'Classify the following {RESPONSE} extracted from a {INTERACTION} with information about',
            f'Classify the following {RESPONSE} of a {INTERACTION} with info about',
            f'Classify the following {RESPONSE} extracted from a {INTERACTION} with info about',
            f'Classify the following {RESPONSE} of a {INTERACTION} with',
            f'Classify the following {RESPONSE} extracted from a {INTERACTION} with',
            f'Annotate the following {RESPONSE} of a {INTERACTION} with information about',
            f'Annotate the following {RESPONSE} extracted from a {INTERACTION} with information about',
            f'Annotate the following {RESPONSE} of a {INTERACTION} with info about',
            f'Annotate the following {RESPONSE} extracted from a {INTERACTION} with info about',
            f'Annotate the following {RESPONSE} of a {INTERACTION} with',
            f'Annotate the following {RESPONSE} extracted from a {INTERACTION} with'
        ]
    }
    DISCRIMINATIVE_TASK_SUFFIX: Dict[str, List[str]] = {
        'global': [''], 'local': ['', f'given the {CONTEXT}']
    }

    GENERATIVE_TASK_STARTER: List[str] = ['', f'The {INTERACTION} begins.']
    DISCRIMINATIVE_TASK_STARTER: Dict[bool, List[str]] = {
        False: ['', 'Label:', 'Annotation:'], True: ['', 'Labels:', 'Annotations:']
    }
    DISCRIMINATIVE_TASK_PROMPT: Dict[str, List[str]] = {
        'global': [
            f'{LABEL_ID}{CLS_SEP}',
            f'{INTERACTION} {DIALOGUE_CHUNK} {LABEL_ID}{CLS_SEP}',
            f'The {LABEL_ID} of the {INTERACTION} {DIALOGUE_CHUNK} is'
        ],
        'local': [f'{LABEL_ID}{CLS_SEP}', f'{RESPONSE} {LABEL_ID}{CLS_SEP}', f'The {LABEL_ID} of the {RESPONSE} is']
    }

    DIALOGUE_CHUNK_SEP: List[str] = ['', f'{DIALOGUE_CHUNK}:', f'{INTERACTION} {DIALOGUE_CHUNK}:']
    DIALOGUE_CONTEXT_SEP: List[str] = [f'{CONTEXT}:', f'{INTERACTION} {CONTEXT}:']
    DIALOGUE_RESPONSE_SEP: List[str] = [f'{RESPONSE}:', f'Follow-up {RESPONSE}:']
    # Labels
    GLOBAL_LABELS_INTRO: str = [
        f'The {INTERACTION} is grounded in',
        f'The {INTERACTION} is labelled with',
        f'The {INTERACTION} is annotated with'
    ]
    LINE_LABELS_INTRO: str = [
        f'Each line of the {INTERACTION} is labelled with',
        f'Each line in the {INTERACTION} is labelled with',
        f'Each turn of the {INTERACTION} is labelled with',
        f'Each turn in the {INTERACTION} is labelled with',
        f'Each utterance of the {INTERACTION} is labelled with',
        f'Each utterance in the {INTERACTION} is labelled with'
    ]
    DESCRIPTION_AND_VALUES_LABEL: List[str] = [
        f'{LABEL_DESCRIPTION}. The possible values are: {LABEL_VALUES}',
        f'{LABEL_DESCRIPTION}. The acceptable values are: {LABEL_VALUES}',
        f'{LABEL_DESCRIPTION}. The values are: {LABEL_VALUES}'
    ]
    DESCRIPTION_ONLY_LABEL: str = [f'{LABEL_DESCRIPTION}']
    VALUES_ONLY_LABEL: List[str] = [
        f'The possible values of {LABEL_ID} are: {LABEL_VALUES}',
        f'The acceptable values of {LABEL_ID} are: {LABEL_VALUES}',
        f'The values of {LABEL_ID} are: {LABEL_VALUES}'
    ]
    # IDs
    HUMAN_SPEAKER_IDS: Dict[str, List[Tuple[str, str]]] = {
        'speaker': [
            ('Speaker A', 'Speaker B'),
            ('Speaker 1', 'Speaker 2'),
            ('Speaker B', 'Speaker A'),
            ('Speaker 2', 'Speaker 1')
        ],
        'person': [
            ('Person A', 'Person B'),
            ('Person 1', 'Person 2'),
            ('Person B', 'Person A'),
            ('Person 2', 'Person 1')
        ],
        'user': [
            ('User A', 'User B'),
            ('User 1', 'User 2'),
            ('User B', 'User A'),
            ('User 2', 'User 1')
        ]
    }
    HUMAN_BOT_SPEAKER_IDS: Dict[str, List[Tuple[str, str]]] = {
        'AI': [
            ('AI', 'Speaker'),
            ('AI', 'Person'),
            ('AI', 'User'),
            ('Speaker', 'AI'),
            ('Person', 'AI'),
            ('User', 'AI')
        ],
        'chatbot': [
            ('Chatbot', 'Speaker'),
            ('Chatbot', 'Person'),
            ('Chatbot', 'User'),
            ('Speaker', 'Chatbot'),
            ('Person', 'Chatbot'),
            ('User', 'Chatbot')
        ]
    }
    # Symbols
    BOL_SYM: List[str] = ['', '- ']
    EOL_SYM: List[str] = ['\n', '\n\n']
    SEP_SYM: List[str] = [':', ' >>>', ' >']
    CLS_SYM: List[str] = [':', ' >>>', ' >', ' =>', ' ->', ' =']
    LBL_BRKS: List[Tuple[str, str]] = [('[', ']'), ('(', ')'), ('{', '}'), ('*', '*')]
    LBL_DESC_SYM: List[Tuple[str, str]] = [('', '. '), ('- ', ';\n'), ('- ', '.\n')]
    LBL_BOL_EOL_SYM: List[Tuple[str, str]] = [('', '.\n'), ('', '. '), ('- ', ';\n'), ('- ', '.\n')]
    GLOBAL_LABELS_METADATA: Optional[Dict[str, Dict[str, Optional[str]]]] = None  # Use placeholder for speaker persona
    LINE_LABELS_METADATA: Optional[Dict[str, Dict[str, Optional[str]]]] = None
    # Probabilities for the composable strings
    CONVERSATION_STARTER_P: List[float] = [0.9, 0.1]
    BOL_SYM_P: List[float] = [0.95, 0.05]
    EOL_SYM_P: List[float] = [0.5, 0.5]
    SEP_SYM_P: List[float] = [0.9, 0.05, 0.05]
    CLS_SYM_P: List[float] = [0.8, 0.04, 0.04, 0.04, 0.04, 0.04]
    LBL_BRKS_P: List[float] = [0.45, 0.45, 0.05, 0.05]
    LBL_DESC_SYM_P: List[float] = [0.4, 0.3, 0.3]
    LBL_BOL_EOL_SYM_P: List[float] = [0.7, 0.1, 0.1, 0.1]
    # Dropout probabilities
    P_DROP_PREMISE: float = 0.5
    P_DROP_TASK: float = 0.1
    P_DROP_LBL_DET: float = 0.1
    P_DROP_LBL_DESC: float = 0.1
    P_DROP_LBL: float = 0.3
    P_DROP_CONTEXT: float = 0.2

    # NOTE during training notation is sampled and shuffled as much as possible

    # TODO prior dropouts
    #   Which labels
    #   if values
    #   which speakers

    # NOTE for evaluation process one label at the time without the others

    def __init__(
            self,
            corpus_dir_path: str,
            split: Literal['train', 'validation', 'test'],
            tokenizer: PreTrainedTokenizer,
            augmentation: Optional[bool] = None,
            dropout: Optional[bool] = None,
            sample: Optional[Union[int, float]] = None,
            max_chunk_turns: Optional[int] = None,
            max_context_turns: Optional[int] = None,
            min_turns: Optional[int] = None,
            joblib_backend: str = 'threading',
            n_jobs: int = -1,
            verbosity_level: int = 0,
            random_seed: Optional[int] = None,
            holdout: Optional[Union[int, float]] = None,
            cache_data_path: Optional[str] = None
    ):
        super(_DialogueCorpus, self).__init__()
        # Path to corpus main directory
        self.corpus_dir_path: str = corpus_dir_path
        # Data split identifier
        self.split: Literal['train', 'validation', 'test'] = split
        # Tokeniser to prepare inputs
        self.tokenizer: PreTrainedTokenizer = tokenizer

        # Set use of noise to change speakers, prompts and labels
        self.augmentation: bool = augmentation if augmentation is not None else self.split == 'train'
        self.dropout: bool = dropout if dropout is not None else self.split == 'train'

        # If to apply random subsampling to the data (useful for testing)
        self.sample: Optional[Union[int, float]] = sample
        # Maximum number of utterances in chunk to classify
        self.max_chunk_turns: Optional[int] = max_chunk_turns
        # Maximum number of utterances in context
        self.max_context_turns: Optional[int] = max_context_turns
        # Minimum number of turns per dialogue
        self.min_turns: Optional[int] = min_turns

        # Parallel backend
        self.joblib_backend: str = joblib_backend
        # Number of concurrent jobs
        self.n_jobs: int = n_jobs
        # Verbosity level
        self.verbosity_level: int = verbosity_level

        # Random seed for random operations
        self.random_seed: Optional[int] = random_seed
        # Size of hold out data to create custom train-validation-test splits
        self.holdout: Optional[Union[int, float]] = holdout

        # Load data
        if cache_data_path is not None:
            self.cache_file_path: os.path.join(cache_data_path, f'{_DialogueCorpus.IDENTIFIER}.pkl')
        else:
            self.cache_file_path = None
        # Load data if already preprocessed
        if self.cache_file_path is not None:
            if os.path.exists(self.cache_file_path):
                with open(self.cache_file_path, 'rb') as f:
                    self.data: List[Dict[str, Union[str, Dict[str, str]]]] = pickle.load(f)
            else:
                self.data: List[Dict[str, Union[str, Dict[str, str]]]] = self._load_samples()
                with open(self.cache_file_path, 'wb') as f:
                    pickle.dump(self.data, f)
        else:
            self.data: List[Dict[str, Union[str, Dict[str, str]]]] = self._load_samples()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Dict[str, Union[str, Dict[str, str]]]:
        return self.data[index]

    @classmethod
    def _preprocess_text(cls, text: str) -> str:
        for u_code_sym, replace_sym in cls.UNICODE_SWITCH_LIST:
            text = text.replace(u_code_sym, replace_sym)  # Taken from ParlAI preprocessing
        text = re.sub(r'\.(\w)', r' . \1', text)  # Taken from ParlAI preprocessing
        text = re.sub('[ \t\n]+', ' ', text)
        text = text.strip()

        return text

    def _preprocess_dialogue(self, *args, **kwargs) -> Dict[str, Union[str, Dict[str, str]]]:
        raise NotImplementedError()

    def _load_samples(self, *args, **kwargs) -> List[Dict[str, Union[str, Dict[str, str]]]]:
        raise NotImplementedError()

    def prepare_dialogue_eval(self, dialogue) -> Dict[str, str]:
        # Get speakers data
        speakers, bot = self._get_speakers(dialogue, False)
        # Get interaction, chunk, context and response types
        interaction, chunk, context, response = self._get_interaction_components(dialogue, False)
        # Get metadata and labels
        global_labels_metadata, line_labels_metadata = self._get_labels_metadata(speakers, dialogue, False, False)
        global_labels, line_labels = self._get_labels(
            dialogue, False, global_labels_metadata, line_labels_metadata
        )
        # Get dialogue lines
        lines = [utterance['text'] for utterance in dialogue['utterances']]
        # Data accumulator
        data = {'generator': dict(), 'discriminator': dict()}
        # Get data for generator model
        data['generator']['plain'] = self._compose_generator_dialogue_eval(speakers, lines, interaction)
        if global_labels_metadata is not None or line_labels_metadata is not None:
            # Get data for conditioned generation
            data['generator']['conditioned'] = self._compose_generator_dialogue_eval(
                speakers,
                lines,
                interaction,
                global_labels_metadata=global_labels_metadata,
                global_labels=global_labels,
                line_labels_metadata=line_labels_metadata,
                line_labels=line_labels
            )
            # Get data for discriminator mode
            for label_type, labels_metadata in zip(('global', 'local'), (global_labels_metadata, line_labels_metadata)):
                if labels_metadata is not None and len(labels_metadata) > 0:
                    for label in labels_metadata:
                        if labels_metadata[label].get('values') is not None and len(labels_metadata[label].get('values')) > 0:
                            if label_type not in data['discriminator']:
                                data['discriminator'][label_type] = dict()
                            if label_type == 'global':
                                tmp_global_labels_metadata = {label: labels_metadata[label]}
                                tmp_local_labels_metadata = None
                            else:
                                tmp_global_labels_metadata = None
                                tmp_local_labels_metadata = {label: labels_metadata[label]}
                            # Get data for discriminator
                            data['discriminator'][label_type][label] = {
                                approach: self._compose_discriminator_dialogue_eval(
                                    label_type,
                                    label,
                                    approach,
                                    self.tokenizer,
                                    speakers,
                                    lines,
                                    interaction,
                                    chunk=chunk,
                                    context=context,
                                    response=response,
                                    global_labels_metadata=tmp_global_labels_metadata,
                                    global_labels=global_labels,
                                    line_labels_metadata=tmp_local_labels_metadata,
                                    line_labels=line_labels,
                                    max_chunk_turns=self.max_chunk_turns,
                                    max_context_turns=self.max_context_turns
                                )
                                for approach in ['posterior', 'infilling', 'prediction']
                            }

        return data

    def prepare_dialogue(
            self,
            dialogue: Dict,
            model_type: Literal['generator', 'discriminator'],
            use_annotations: bool,
            augmentation: bool,
            dropout: bool
    ) -> List[str]:
        # Get speakers data
        speakers, bot = self._get_speakers(dialogue, augmentation)
        # Get interaction, chunk, context and response types
        interaction, chunk, context, response = self._get_interaction_components(dialogue, augmentation)
        # Get metadata and labels (if not plaintext)
        if (use_annotations or model_type == 'discriminator') and (self.GLOBAL_LABELS_METADATA is not None or self.LINE_LABELS_METADATA is not None):
            global_labels_metadata, line_labels_metadata = self._get_labels_metadata(speakers, dialogue, augmentation, dropout)
            global_labels, line_labels = self._get_labels(
                dialogue, augmentation, global_labels_metadata, line_labels_metadata
            )
        else:
            global_labels_metadata = line_labels_metadata = global_labels = line_labels = None
        # Get dialogue lines
        lines = [utterance['text'] for utterance in dialogue['utterances']]
        # Depending on model type generate data
        if model_type == 'generator':
            # Prepare dialogue sequences
            seqs = self._compose_generator_dialogue(
                self.tokenizer,
                speakers,
                lines,
                interaction,
                augmentation,
                dropout,
                bot,
                global_labels_metadata=global_labels_metadata,
                global_labels=global_labels,
                line_labels_metadata=line_labels_metadata,
                line_labels=line_labels,
                min_turns=self.min_turns
            )
        elif model_type == 'discriminator':
            # Prepare dialogue for global labels classification
            if global_labels_metadata is not None and len(global_labels_metadata) > 0:
                #
                seqs_global_labels = self._compose_discriminator_dialogue(
                    'global',
                    self.tokenizer,
                    speakers,
                    lines,
                    interaction,
                    augmentation,
                    dropout,
                    bot,
                    chunk=chunk,
                    tgt_global_labels_metadata=global_labels_metadata,
                    tgt_global_labels=global_labels,
                    line_labels_metadata=line_labels_metadata if use_annotations else None,
                    line_labels=line_labels if use_annotations else None,
                    max_chunk_turns=self.max_chunk_turns
                )
            else:
                seqs_global_labels = []
            # Line labels
            if line_labels_metadata is not None and len(line_labels_metadata) > 0:
                #
                seqs_local_labels = self._compose_discriminator_dialogue(
                    'local',
                    self.tokenizer,
                    speakers,
                    lines,
                    interaction,
                    augmentation,
                    dropout,
                    bot,
                    context=context,
                    response=response,
                    global_labels_metadata=global_labels_metadata if use_annotations else None,
                    global_labels=global_labels if use_annotations else None,
                    line_labels_metadata=line_labels_metadata if use_annotations else None,
                    line_labels=line_labels if use_annotations else None,
                    tgt_line_labels_metadata=line_labels_metadata,
                    tgt_line_labels=line_labels,
                    max_context_turns=self.max_context_turns
                )
            else:
                seqs_local_labels = []
            seqs = seqs_global_labels + seqs_local_labels
        else:
            raise ValueError(f'Unknown model type: \'{model_type}\'')

        return seqs

    def get_dialogues(self, *args, **kwargs) -> List[str]:
        # Preprocess and prepare dialogues
        with parallel_backend(self.joblib_backend, n_jobs=self.n_jobs):
            return sum(Parallel(verbose=self.verbosity_level)(
                delayed(self.prepare_dialogue)(dialogue, *args, **kwargs) for dialogue in self.data
            ), [])

    def get_dialogues_eval(self):
        # Preprocess and prepare dialogues
        with parallel_backend(self.joblib_backend, n_jobs=self.n_jobs):
            return Parallel(verbose=self.verbosity_level)(
                delayed(self.prepare_dialogue_eval)(dialogue) for dialogue in self.data
            )

    def get_data_for_fitting(
            self,
            full: bool = True,
            plaintext: bool = True,
            dropout: bool = True,
            augmentation: bool = True,
            generator: bool = True,
            discriminator: bool = True
    ) -> List[str]:
        assert generator or discriminator
        assert full or plaintext or augmentation
        #
        samples = []
        #
        if full and (self.GLOBAL_LABELS_METADATA is not None or self.LINE_LABELS_METADATA is not None):
            if generator:
                samples += self.get_dialogues('generator', True, augmentation, False)
            if discriminator:
                samples += self.get_dialogues('discriminator', True, augmentation, False)
        if plaintext:
            if generator:
                samples += self.get_dialogues('generator', False, augmentation, False)
            if discriminator:
                samples += self.get_dialogues('discriminator', False, augmentation, False)
        if dropout:
            if generator:
                samples += self.get_dialogues('generator', True, augmentation, dropout)
            if discriminator:
                samples += self.get_dialogues('discriminator', True, augmentation, dropout)

        return samples

    def get_data_for_evaluation(self):
        samples = self.get_dialogues_eval()

        samples = {
            'generator': {
                'plain': [sample['generator']['plain'] for sample in samples],
                'conditioned': [sample['generator']['conditioned'] for sample in samples] if 'conditioned' in samples[0]['generator'] else None
            },
            'discriminator': {
                'global': {
                    lbl: {
                        'posterior':  [sample['discriminator']['global'][lbl]['posterior'] for sample in samples],
                        'infilling':  [sample['discriminator']['global'][lbl]['infilling'] for sample in samples],
                        'prediction': [sample['discriminator']['global'][lbl]['prediction'] for sample in samples]
                    }
                    for lbl in samples[0]['discriminator']['global']
                } if 'global' in samples[0]['discriminator'] else None,
                'local': {
                    lbl: {
                        'posterior':  [sample['discriminator']['local'][lbl]['posterior'] for sample in samples],
                        'infilling':  [sample['discriminator']['local'][lbl]['infilling'] for sample in samples],
                        'prediction': [sample['discriminator']['local'][lbl]['prediction'] for sample in samples]
                    }
                    for lbl in samples[0]['discriminator']['local']
                } if 'local' in samples[0]['discriminator'] else None
            }
        }

        return samples

    @classmethod
    def _get_speakers(
            cls, dialogue: Dict, augmentation: bool
    ) -> Tuple[Tuple[str, str], bool]:
        if augmentation:
            bot = random.choice([True, False])
            if bot:
                speakers = random.choice(cls.HUMAN_BOT_SPEAKER_IDS[random.choice(list(cls.HUMAN_BOT_SPEAKER_IDS))])
            else:
                speakers = random.choice(cls.HUMAN_SPEAKER_IDS[random.choice(list(cls.HUMAN_SPEAKER_IDS))])
        else:
            speakers = cls.HUMAN_SPEAKER_IDS['speaker'][0]
            bot = False

        return speakers, bot

    @classmethod
    def _get_interaction_components(
            cls, dialogue: Dict, augmentation: bool
    ) -> Tuple[str, str, str, str]:
        # Interaction
        interaction = random.choice(cls.INTERACTION_TYPE) if augmentation else cls.INTERACTION_TYPE[0]
        # Chunk
        chunk = random.choice(cls.CHUNK_TYPE) if augmentation else cls.CHUNK_TYPE[0]
        # Context
        context = random.choice(cls.CONTEXT_TYPE) if augmentation else cls.CONTEXT_TYPE[0]
        # Response
        response = random.choice(cls.RESPONSE_TYPE) if augmentation else cls.RESPONSE_TYPE[0]

        return interaction, chunk, context, response

    @classmethod
    def _prepare_labels_metadata(cls, labels_metadata: Dict, augmentation: bool, dropout: List[bool]) -> Dict:
        labels_metadata = labels_metadata.copy()
        if augmentation:
            labels_metadata = {
                lbl: labels_metadata[lbl]
                for lbl, drop in zip(random.sample(list(labels_metadata), len(labels_metadata)), dropout)
                if not drop
            }
        for lbl in labels_metadata:
            labels_metadata[lbl]['use_context'] = (
                True if not augmentation else random.uniform(0.0, 1.0) > cls.P_DROP_CONTEXT
            )

        return labels_metadata

    @classmethod
    def _get_labels_metadata(
            cls, speakers: Tuple[str, str], dialogue: Dict, augmentation: bool, dropout: bool
    ) -> Tuple[Dict[str, str], Dict[str, str]]:
        assert not dropout or augmentation
        # Additional parameters
        n_global_labels = len(cls.GLOBAL_LABELS_METADATA) if cls.GLOBAL_LABELS_METADATA is not None else 0
        n_line_labels = len(cls.LINE_LABELS_METADATA) if cls.LINE_LABELS_METADATA is not None else 0
        n_labels = n_global_labels + n_line_labels
        if dropout:
            lbl_dropout = random.sample(([True] * (n_labels - 1)) + ([False] * n_labels), n_labels)
            lbl_dropout = [d and random.uniform(0.0, 1.0) > cls.P_DROP_LBL for d in lbl_dropout]
        else:
            lbl_dropout = [False] * n_labels
        # Get global labels
        if cls.GLOBAL_LABELS_METADATA is not None:
            global_labels_metadata = cls._prepare_labels_metadata(
                cls.GLOBAL_LABELS_METADATA, augmentation, lbl_dropout[:n_global_labels]
            )
        else:
            global_labels_metadata = None
        # Get line labels
        if cls.LINE_LABELS_METADATA is not None:
            line_labels_metadata = cls._prepare_labels_metadata(
                cls.LINE_LABELS_METADATA, augmentation, lbl_dropout[-n_line_labels:]
            )
        else:
            line_labels_metadata = None

        return global_labels_metadata, line_labels_metadata

    @classmethod
    def _get_labels(
            cls,
            dialogue: Dict,
            augmentation: bool,
            global_labels_metadata: Optional[Dict[str, str]],
            line_labels_metadata: Optional[Dict[str, str]]
    ) -> Tuple[Optional[Dict[str, str]], Optional[List[Dict[str, str]]]]:
        return None, None

    @classmethod
    def _compose_premise(cls, speakers: Tuple[str, str], interaction: str, augmentation: bool, bot: bool) -> str:
        # Presentation line
        presentation = random.choice(cls.PRESENTATION) if augmentation else cls.PRESENTATION[0]
        # Generate speakers introduction and description
        if bot:
            sp1, sp2 = speakers
            if any(sp1.lower() in key.lower() or key.lower() in sp1.lower() for key in cls.HUMAN_BOT_SPEAKER_IDS):
                p1_list, p2_list = cls.BOT_PARTICIPANT, cls.HUMAN_PARTICIPANT
            else:
                p1_list, p2_list = cls.HUMAN_PARTICIPANT, cls.BOT_PARTICIPANT
            p1 = random.choice(p1_list) if augmentation else p1_list[0]
            p2 = random.choice(p2_list) if augmentation else p2_list[0]
            speakers_description = f'{p1}, {presentation} {sp1}, and {p2}, {presentation} {sp2}'
        else:
            speakers = sorted(speakers)
            sp1, sp2 = speakers
            ps = random.choice(cls.HUMAN_PARTICIPANTS) if augmentation else cls.HUMAN_PARTICIPANTS[0]
            speakers_description = f'{ps}, {presentation} {sp1} and {sp2}'
        # Compose premise
        premise = random.choice(cls.PREMISE) if augmentation else cls.PREMISE[0]
        premise = premise.replace(INTERACTION, interaction)
        premise = premise.replace(SPEAKER_DESCRIPTIONS, speakers_description)

        return premise

    @classmethod
    def _compose_label_details(
            cls,
            label: str,
            label_description: str,
            label_values: Optional[List[str]],
            augmentation: bool,
            dropout: bool,
            bol: str = ''
    ) -> str:
        assert not dropout or augmentation
        # Select base string
        if label_values is not None:
            if dropout and random.uniform(0.0, 1.0) < cls.P_DROP_LBL_DET:
                if augmentation:
                    lbl_details = random.choice(cls.DESCRIPTION_AND_VALUES_LABEL)
                else:
                    lbl_details = cls.DESCRIPTION_AND_VALUES_LABEL[0]
            else:
                lbl_details = random.choice([
                    random.choice(cls.DESCRIPTION_AND_VALUES_LABEL),
                    random.choice(cls.DESCRIPTION_ONLY_LABEL),
                    random.choice(cls.VALUES_ONLY_LABEL)
                ])
        else:
            if augmentation:
                lbl_details = random.choice(cls.DESCRIPTION_ONLY_LABEL)
            else:
                lbl_details = cls.DESCRIPTION_ONLY_LABEL[0]
        # Replace label specific details
        lbl_details = lbl_details.replace(LABEL_ID, label)
        lbl_details = lbl_details.replace(LABEL_DESCRIPTION, label_description)
        if label_values is not None:
            lbl_details = lbl_details.replace(
                LABEL_VALUES,
                ', '.join(random.sample(label_values, len(label_values)) if augmentation else label_values)
            )
        # Compose details string
        details = f'{bol}{lbl_details}'

        return details

    @classmethod
    def _get_label_descriptions(
            cls,
            augmentation: bool,
            dropout: bool,
            global_labels_metadata: Optional[Dict] = None,
            line_labels_metadata: Optional[Dict] = None,
            bol: str = ''
    ) -> List[str]:
        assert not dropout or augmentation
        # Available notations
        if global_labels_metadata is not None and line_labels_metadata is not None:
            labels_metadata = [global_labels_metadata, line_labels_metadata]
        elif global_labels_metadata is not None and line_labels_metadata is None:
            labels_metadata = [global_labels_metadata]
        elif global_labels_metadata is None and line_labels_metadata is not None:
            labels_metadata = [line_labels_metadata]
        else:
            labels_metadata = []
        if augmentation:
            random.shuffle(labels_metadata)
        # Get list of label descriptions
        descriptions = [
            cls._compose_label_details(
                metadata[label]['id'],
                metadata[label]['description'],
                metadata[label].get('values'),
                augmentation,
                dropout,
                bol=bol
            )
            for metadata in labels_metadata for label in metadata
            if metadata is not None and not (dropout and random.uniform(0.0, 1.0) < cls.P_DROP_LBL_DESC)
        ]

        return descriptions

    @classmethod
    def _compose_label_descriptions(
            cls,
            interaction: str,
            augmentation: bool,
            dropout: bool,
            global_labels_metadata: Optional[Dict] = None,
            line_labels_metadata: Optional[Dict] = None
    ) -> str:
        # Additional parameters
        bol, eol = random.choice(cls.LBL_DESC_SYM) if augmentation else cls.LBL_DESC_SYM[0]
        # Compose labels presentation
        if global_labels_metadata is not None and len(global_labels_metadata) > 0:
            global_labels_intro = random.choice(cls.GLOBAL_LABELS_INTRO) if augmentation else cls.GLOBAL_LABELS_INTRO[0]
            global_labels_intro = global_labels_intro.replace(INTERACTION, interaction)
            tmp_lbl_keys = list(global_labels_metadata)
            if len(global_labels_metadata) > 1:
                global_labels_intro = f'{global_labels_intro} the ' \
                                      f'{", the".join(global_labels_metadata[lbl]["id"] for lbl in tmp_lbl_keys[:-1])} ' \
                                      f'and the {global_labels_metadata[tmp_lbl_keys[-1]]["id"]}.'
            else:
                global_labels_intro = f'{global_labels_intro} the ' \
                                      f'{global_labels_metadata[tmp_lbl_keys[0]]["id"]}.'
        else:
            global_labels_intro = ''
        if line_labels_metadata is not None and len(line_labels_metadata) > 0:
            line_labels_intro = random.choice(cls.LINE_LABELS_INTRO) if augmentation else cls.LINE_LABELS_INTRO[0]
            line_labels_intro = line_labels_intro.replace(INTERACTION, interaction)
            tmp_lbl_keys = list(line_labels_metadata)
            if len(line_labels_metadata) > 1:
                line_labels_intro = f'{line_labels_intro} the ' \
                                    f'{", the".join(line_labels_metadata[lbl]["id"] for lbl in tmp_lbl_keys[:-1])} ' \
                                    f'and the {line_labels_metadata[tmp_lbl_keys[-1]]["id"]}.'
            else:
                line_labels_intro = f'{line_labels_intro} the ' \
                                    f'{line_labels_metadata[tmp_lbl_keys[0]]["id"]}.'
        else:
            line_labels_intro = ''
        # Compose description of labels
        descriptions = eol.join(cls._get_label_descriptions(
            augmentation,
            dropout,
            global_labels_metadata=global_labels_metadata,
            line_labels_metadata=line_labels_metadata,
            bol=bol
        ))
        descriptions = f'{global_labels_intro} {line_labels_intro}\n{descriptions}.'.strip()

        return descriptions

    @classmethod
    def _compose_task_info(
            cls,
            speakers: Tuple[str, str],
            model_type: Literal['generator', 'discriminator'],
            augmentation: bool,
            tgt_labels_metadata: Optional[Dict] = None,
            label_type: Optional[Literal['global', 'local']] = None,
            interaction: Optional[str] = None,
            chunk: Optional[str] = None,
            context: Optional[str] = None,
            response: Optional[str] = None
    ) -> str:
        # Compose task info depending on the task
        if model_type == 'generator':
            # Additional parameters
            sp1, sp2 = speakers
            # Compose string
            if augmentation:
                task_info = random.choice(cls.GENERATIVE_TASK_PREFIX)
            else:
                task_info = cls.GENERATIVE_TASK_PREFIX[0]
            task_info = task_info.replace(SPEAKERS, f'{sp1} and {sp2}')
        elif model_type == 'discriminator':
            if label_type in cls.DISCRIMINATIVE_TASK_PREFIX:
                # Additional parameters
                if augmentation:
                    task_prefix = random.choice(cls.DISCRIMINATIVE_TASK_PREFIX[label_type])
                    task_suffix = random.choice(cls.DISCRIMINATIVE_TASK_SUFFIX[label_type])
                else:
                    task_prefix = cls.DISCRIMINATIVE_TASK_PREFIX[label_type][0]
                    task_suffix = cls.DISCRIMINATIVE_TASK_SUFFIX[label_type][0]
                tmp_lbl_keys = list(tgt_labels_metadata)
                if len(tgt_labels_metadata) > 1:
                    task_lbls = f'the {", the".join(tgt_labels_metadata[lbl]["id"] for lbl in tmp_lbl_keys[:-1])} ' \
                                f'and the {tgt_labels_metadata[tmp_lbl_keys[-1]]["id"]}'
                else:
                    task_lbls = f'the {tgt_labels_metadata[tmp_lbl_keys[0]]["id"]}'
                # Compose string
                task_info = f'{task_prefix} {task_lbls} {task_suffix}'
                if label_type == 'global':
                    task_info = task_info.replace(INTERACTION, interaction)
                    task_info = task_info.replace(DIALOGUE_CHUNK, chunk)
                elif label_type == 'local':
                    task_info = task_info.replace(INTERACTION, interaction)
                    if tgt_labels_metadata.get('use_context', True):
                        task_info = task_info.replace(CONTEXT, context)
                    elif len(task_suffix) > 0:
                        task_info = task_info.replace(task_suffix, '')
                    task_info = task_info.replace(RESPONSE, response)
                task_info = task_info.strip() + '.'
            else:
                raise ValueError(
                    f'Unknown label type: {label_type}. Accepted model types are \'global\' and \'local\'.'
                )
        else:
            raise ValueError(
                f'Unknown model type: {model_type}. Accepted model types are \'generator\' and \'discriminator\'.'
            )

        return task_info

    @classmethod
    def _compose_task_details(
            cls,
            speakers: Tuple[str, str],
            model_type: Literal['generator', 'discriminator'],
            interaction: str,
            augmentation: bool,
            dropout: bool,
            global_labels_metadata: Optional[Dict] = None,
            line_labels_metadata: Optional[Dict] = None,
            tgt_labels_metadata: Optional[Dict] = None,
            label_type: Optional[Literal['global', 'local']] = None,
            chunk: Optional[str] = None,
            context: Optional[str] = None,
            response: Optional[str] = None
    ) -> str:
        assert not dropout or augmentation
        # Additional parameters
        compose_descriptions = not (
                (
                        (global_labels_metadata is None or len(global_labels_metadata) == 0) and
                        (line_labels_metadata is None or len(line_labels_metadata) == 0)
                ) or (dropout and random.uniform(0.0, 1.0) < cls.P_DROP_PREMISE)
        )

        # Task description
        task_info = cls._compose_task_info(
            speakers,
            model_type,
            augmentation,
            tgt_labels_metadata=tgt_labels_metadata,
            label_type=label_type,
            interaction=interaction,
            chunk=chunk,
            context=context,
            response=response
        )
        # Compose label info (in any)
        if compose_descriptions:
            lbl_info = cls._compose_label_descriptions(
                interaction,
                augmentation,
                dropout,
                global_labels_metadata=global_labels_metadata,
                line_labels_metadata=line_labels_metadata
            )
        else:
            lbl_info = ''

        # Compose task details
        task_details = f'{task_info}\n{lbl_info}'.strip()

        return task_details

    @classmethod
    def _compose_task_description(
            cls,
            speakers: Tuple[str, str],
            model_type: Literal['generator', 'discriminator'],
            interaction: str,
            augmentation: bool,
            dropout: bool,
            bot: bool,
            global_labels_metadata: Optional[Dict] = None,
            line_labels_metadata: Optional[Dict] = None,
            tgt_labels_metadata: Optional[Dict] = None,
            label_type: Optional[Literal['global', 'local']] = None,
            chunk: Optional[str] = None,
            context: Optional[str] = None,
            response: Optional[str] = None
    ) -> str:
        assert not dropout or augmentation
        # Compose premise
        if dropout and random.uniform(0.0, 1.0) < cls.P_DROP_PREMISE:
            premise = ''
        else:
            premise = cls._compose_premise(speakers, interaction, augmentation, bot)
        # Compose task details
        if dropout and random.uniform(0.0, 1.0) < cls.P_DROP_TASK:
            task_details = ''
        else:
            task_details = cls._compose_task_details(
                speakers,
                model_type,
                interaction,
                augmentation,
                dropout,
                global_labels_metadata=global_labels_metadata,
                line_labels_metadata=line_labels_metadata,
                tgt_labels_metadata=tgt_labels_metadata,
                label_type=label_type,
                chunk=chunk,
                context=context,
                response=response
            )
        # Compose task description
        task_description = f'{premise}\n\n{task_details}'.strip()

        return task_description

    @classmethod
    def _compose_line(
            cls,
            line: str,
            speaker: str,
            bol: str,
            sep: str,
            eol: str,
            labels: Optional[Dict[str, str]] = None,
            line_labels_metadata: Optional[List[str]] = None,
            lbl_brackets: Optional[Tuple[str, str]] = None,
            split_lines: bool = False
    ) -> Union[str, Tuple[str, str, str]]:
        assert (labels is None and line_labels_metadata is None and lbl_brackets is None) or \
               (labels is not None and line_labels_metadata is not None and lbl_brackets is not None)
        key_id = 'id'
        if labels is not None and len(line_labels_metadata) > 0:
            lbl_l_bracket, lbl_r_bracket = lbl_brackets
            lbl = f' {lbl_l_bracket}{", ".join(f"{line_labels_metadata[lbl][key_id]}: {labels[lbl]}" for lbl in line_labels_metadata)}{lbl_r_bracket}'
        else:
            lbl = ''
        # Compose dialogue line
        dialogue_line = f'{bol}{speaker}{lbl}{sep} {line}{eol}'
        if split_lines:
            prefix = f'{bol}{speaker}'
            prompt = f'{bol}{speaker}{lbl}{sep}'
            return prefix, prompt, dialogue_line
        else:
            return dialogue_line

    @classmethod
    def _get_dialogue_lines(
            cls,
            lines: List[str],
            speakers: Tuple[str, str],
            augmentation: bool,
            dropout: bool,
            bol: Optional[str] = None,
            sep: Optional[str] = None,
            eol: Optional[str] = None,
            eos: Optional[str] = None,
            labels: Optional[List[Dict[str, str]]] = None,
            line_labels_metadata: Optional[List[str]] = None,
            lbl_brackets: Optional[Tuple[str, str]] = None,
            split_lines: bool = False
    ) -> Union[List[str], List[Tuple[str, str, str]]]:
        assert (labels is None and line_labels_metadata is None) or (labels is not None and line_labels_metadata is not None)
        assert not dropout or augmentation
        # Additional parameters
        if augmentation:
            if bol is None:
                bol = random.choices(cls.BOL_SYM, weights=cls.BOL_SYM_P)[0]
            if sep is None:
                sep = random.choices(cls.SEP_SYM, weights=cls.SEP_SYM_P)[0]
            if eol is None:
                eol = random.choices(cls.EOL_SYM, weights=cls.EOL_SYM_P)[0]
        else:
            if bol is None:
                bol = cls.BOL_SYM[0]
            if sep is None:
                sep = cls.SEP_SYM[0]
            if eol is None:
                eol = cls.EOL_SYM[0]
        if eos is None:
            eos = eol
        line_terminators = ([eol] * (len(lines) - 1)) + [eos]
        if labels is not None:
            ''' 
            if dropout and random.uniform(0.0, 1.0) < cls.P_DROP_LBL_DET:
            # NOTE cannot drop lbl here because of possible description in
                shift = random.randint(0, 1)
                labels = [lbl if (idx + shift) % 2 == 0 else None for idx, lbl in enumerate(labels)] 
            '''  # TODO decide if keep
            if lbl_brackets is None:
                if augmentation:
                    lbl_brackets = random.choices(cls.LBL_BRKS, weights=cls.LBL_BRKS_P)[0]
                else:
                    lbl_brackets = cls.LBL_BRKS[0]
        else:
            labels = repeat(labels)
        # Compose list of lines
        lines = [
            cls._compose_line(
                line, speaker, bol, sep, eol, lbl, line_labels_metadata, lbl_brackets, split_lines=split_lines
            )
            for speaker, lbl, line, eol in zip(cycle(speakers), labels, lines, line_terminators)
        ]

        return lines

    @classmethod
    def _compose_label_line(
            cls,
            labelling_prompt: str,
            label: str,
            value: str,
            bol: str,
            sep: str,
            eol: str,
            label_type: Literal['global', 'local'],
            interaction: str,
            chunk: Optional[str] = None,
            response: Optional[str] = None,
    ) -> str:
        # Compose string
        label_line = f'{bol}{labelling_prompt} {value}{eol}'
        label_line = label_line.replace(LABEL_ID, label)
        label_line = label_line.replace(CLS_SEP, sep)
        label_line = label_line.replace(INTERACTION, interaction)
        if label_type == 'global':
            label_line = label_line.replace(DIALOGUE_CHUNK, chunk)
        elif label_type == 'local':
            label_line = label_line.replace(RESPONSE, response)
        else:
            raise ValueError(f'Unknown label type: {label_type}. Accepted label types are \'global\' and \'local\'.')
        label_line = capitalize(label_line)

        return label_line

    @classmethod
    def _get_dialogue_labels(
            cls,
            labels: Dict[str, str],
            labels_metadata: Dict[str, Dict[str, str]],
            augmentation: bool,
            label_type: Literal['global', 'local'],
            interaction: str,
            bol: Optional[str] = None,
            sep: Optional[str] = None,
            eol: Optional[str] = None,
            chunk: Optional[str] = None,
            response: Optional[str] = None,
    ) -> List[str]:
        # Additional parameters
        if label_type in cls.DISCRIMINATIVE_TASK_PROMPT:
            if augmentation:
                labelling_prompt = random.choice(cls.DISCRIMINATIVE_TASK_PROMPT[label_type])
            else:
                labelling_prompt = cls.DISCRIMINATIVE_TASK_PROMPT[label_type][0]
        else:
            raise ValueError(f'Unknown label type: {label_type}. Accepted model types are \'global\' and \'local\'.')
        if augmentation:
            if bol is None and eol is None:
                bol, eol = random.choices(cls.LBL_BOL_EOL_SYM, weights=cls.LBL_BOL_EOL_SYM_P)[0]
            if sep is None:
                sep = random.choices(cls.CLS_SYM, weights=cls.CLS_SYM_P)[0]
        else:
            if bol is None and eol is None:
                bol, eol = cls.LBL_BOL_EOL_SYM[0]
            if sep is None:
                sep = cls.CLS_SYM[0]
        eol_list = ([eol] * (len(labels) - 1)) + ['.']
        # Compose labels
        dialogue_labels = [
            cls._compose_label_line(
                labelling_prompt, labels_metadata[lbl]['id'], labels[lbl], bol, sep, eol, label_type, interaction,
                chunk=chunk, response=response
            )
            for lbl, eol in zip(labels_metadata, eol_list)
        ]

        return dialogue_labels

    @classmethod
    def _compose_data_global_label(
            cls,
            tokenizer: PreTrainedTokenizer,
            len_prefix_tok: int,
            len_suffix_tok: int,
            chunk_intro: str,
            lines: List[Tuple[str, str, str]],
            max_chunk_turns: int
    ) -> List[str]:
        # Get token lengths
        len_start_sep_tok = len(tokenizer('\n\n').input_ids)
        len_end_sep_tok = len(tokenizer('\n\n').input_ids)
        # Accumulators for dialogue chunks
        chunks = []
        #
        if len(chunk_intro) > 0:
            chunk_intro += '\n\n'
        chunk_intro = capitalize(chunk_intro)
        # Indices generator
        idxs = (
            (i, i + max_chunk_turns) if i + max_chunk_turns <= len(lines) else (max(0, len(lines) - max_chunk_turns), len(lines))
            for i in range(0, len(lines), max_chunk_turns)
        )
        # Max len admissible
        max_len_tok = tokenizer.model_max_length - (len_prefix_tok + len_start_sep_tok + len_end_sep_tok + len_suffix_tok)
        # Iterate over indices
        for s_idx, e_idx in idxs:
            curr_chunk_lines = lines[s_idx:e_idx]
            # Current chunk
            chunk = chunk_intro + ''.join(line for *_, line in curr_chunk_lines).strip()
            # While the current chunk does not fit in memory iteratively remove line pieces
            while len(tokenizer(chunk).input_ids) > max_len_tok and len(curr_chunk_lines) > 0:
                # Take out first line from context
                prefix, prompt, dialogue_line = curr_chunk_lines.pop(0)
                # Try to remove first half and put it back
                if not dialogue_line[len(prompt):].startswith(' ... '):
                    dialogue_line = f'{prompt} ... {dialogue_line[len(prompt) + ((len(dialogue_line) - len(prompt)) // 2):]}'
                    curr_chunk_lines = [dialogue_line] + curr_chunk_lines
                # Update chunk
                chunk = ''.join(line for *_, line in curr_chunk_lines).strip()
            # Truncate response if still out of boundaries
            chunk = tokenizer.decode(tokenizer(chunk).input_ids[:max_len_tok])
            # Append string to accumulator
            chunks.append(f'\n\n{chunk}\n\n')

        return chunks

    @classmethod
    def _compose_data_local_label(
            cls,
            tokenizer: PreTrainedTokenizer,
            len_prefix_tok: int,
            len_suffix_tok: List[int],
            response_intro: str,
            response_lines: List[str],
            context_intro: Optional[str] = None,
            context_lines: Optional[List[Tuple[str, str, str]]] = None,
            max_context_turns: Optional[int] = None
    ) -> List[str]:
        # Get token lengths
        len_start_sep_tok = len(tokenizer('\n\n').input_ids)
        len_end_sep_tok = len(tokenizer('\n\n').input_ids)
        # Accumulators for context response pairs
        pairs = []
        # Iterate over response to label
        for idx, (response, curr_len_suffix_tok) in enumerate(zip(response_lines, len_suffix_tok)):
            # Max len admissible
            max_len_tok = tokenizer.model_max_length - (len_prefix_tok + len_start_sep_tok + len_end_sep_tok + curr_len_suffix_tok)
            # Additional parameters
            curr_context_lines = context_lines[:idx][-max_context_turns:] if context_lines is not None else None
            if curr_context_lines is not None and len(curr_context_lines) > 0:
                curr_context_intro = context_intro
                context = ''.join(line for *_, line in curr_context_lines).strip()
            else:
                curr_context_intro = ''
                context = ''
            # Compose current pair
            pair = f'{capitalize(curr_context_intro)}\n\n{context}\n\n{capitalize(response_intro)}\n\n{response}'.strip()
            # While the current pair does not fit in memory iteratively remove context pieces
            while len(tokenizer(pair).input_ids) > max_len_tok and curr_context_lines is not None and len(curr_context_lines) > 0:
                # Take out first line from context
                prefix, prompt, dialogue_line = curr_context_lines.pop(0)
                # Try to remove first half and put it back
                if not dialogue_line[len(prompt):].startswith(' ... '):
                    dialogue_line = f'{prompt} ... {dialogue_line[len(prompt) + ((len(dialogue_line) - len(prompt)) // 2):]}'
                    curr_context_lines = [dialogue_line] + curr_context_lines
                # Update pair
                if curr_context_lines is not None and len(curr_context_lines) > 0:
                    curr_context_intro = context_intro
                    context = ''.join(line for *_, line in curr_context_lines).strip()
                else:
                    curr_context_intro = ''
                    context = ''
                pair = f'{capitalize(curr_context_intro)}\n\n{context}\n\n{capitalize(response_intro)}\n\n{response}'.strip()
            # Truncate response if still out of boundaries
            pair = tokenizer.decode(tokenizer(pair).input_ids[:max_len_tok])
            # Append string to accumulator
            pairs.append(f'\n\n{pair}\n\n')

        return pairs

    @classmethod
    def _compose_data_dialogue(
            cls, tokenizer: PreTrainedTokenizer, lines: List[str], len_prefix_tok: int, min_turns: int
    ) -> List[str]:
        # Get token lengths
        len_start_sep_tok = len(tokenizer('\n\n').input_ids)
        len_mid_sep_tok = len(tokenizer('\n\n...\n\n').input_ids)
        # Initialisation
        len_curr_split_tok = len_prefix_tok + len_start_sep_tok
        len_curr_split_turns = 0
        splits = ['\n\n']
        # Loop over dialogue lines
        for line in lines:
            # Get current line length in tokens
            len_line_tok = len(tokenizer(line).input_ids)
            # Append current line
            splits[-1] += line
            # Update counters
            len_curr_split_tok += len_line_tok
            len_curr_split_turns += 1
            # In case the maximum length is reached, start new sequence
            if len_curr_split_tok > tokenizer.model_max_length:
                # Crop latest string at maximum length
                splits[-1] = tokenizer.decode(
                    tokenizer(splits[-1]).input_ids[:tokenizer.model_max_length - len_prefix_tok]
                )
                # Get length on newly started string including latest lines
                len_curr_split_tok = len_prefix_tok + len_mid_sep_tok + len_line_tok
                # Start new split with latest line if not too long
                splits.append(
                    '\n\n...\n\n' + (line if len_curr_split_tok < tokenizer.model_max_length else '')
                )
                # Add latest turn if not too long
                len_curr_split_turns = 1 if len_curr_split_tok < tokenizer.model_max_length else 0
                # Remove tokens of latest line if it was too long
                if len_curr_split_tok < tokenizer.model_max_length:
                    len_curr_split_tok = len_prefix_tok + len_mid_sep_tok
        # Postprocessing (remove dialogue pieces below a certain turn-length threshold)
        if len(splits) > 1 and min_turns is not None and len_curr_split_turns < min_turns:
            splits.pop(-1)

        return splits

    @classmethod
    def _get_global_labels(
            cls, global_labels: Optional[Dict[str, str]], global_labels_metadata: Optional[Dict[str, Dict[str, str]]]
    ) -> List[str]:
        # Compose labels
        labels = [
            capitalize(f'{global_labels_metadata[lbl]["id"]}: {global_labels[lbl]}.')
            for lbl in global_labels_metadata
        ] if global_labels_metadata is not None and len(global_labels_metadata) > 0 else []

        return labels
    
    @classmethod
    def _compose_dialogue_starter(
            cls,
            model_type: Literal['discriminator', 'generator'],
            augmentation: bool,
            interaction: str,
            global_labels: Optional[Dict[str, str]] = None,
            global_labels_metadata: Optional[Dict[str, Dict[str, str]]] = None
    ) -> str:
        # Additional parameters
        global_labels = '\n'.join(label for label in cls._get_global_labels(global_labels, global_labels_metadata))
        if model_type == 'generator':
            # Additional parameters
            if augmentation:
                dialogue_starter = random.choices(cls.GENERATIVE_TASK_STARTER, weights=cls.CONVERSATION_STARTER_P)[0]
            else:
                dialogue_starter = cls.GENERATIVE_TASK_STARTER[0]
        elif model_type == 'discriminator':
            dialogue_starter = ''
        else:
            raise ValueError(
                f'Unknown model type: {model_type}. Accepted model types are \'generator\' and \'discriminator\'.'
            )
        dialogue_starter = capitalize(dialogue_starter.replace(INTERACTION, interaction))
        # Compose starter
        starter = f'{dialogue_starter}\n\n{global_labels}'.strip()
        if len(starter) > 0:
            starter = '\n\n' + starter

        return starter

    @classmethod
    def _compose_discriminator_task(
            cls,
            labels: Union[Dict[str, str], List[Dict[str, str]]],
            labels_metadata: Dict[str, Dict[str, str]],
            augmentation: bool,
            label_type: Literal['global', 'local'],
            eos: str,
            interaction: str,
            chunk: Optional[str] = None,
            response: Optional[str] = None
    ) -> str:
        # Additional parameters
        if augmentation:
            task_starter = random.choice(cls.DISCRIMINATIVE_TASK_STARTER[len(labels) > 1])
        else:
            task_starter = cls.DISCRIMINATIVE_TASK_STARTER[len(labels) > 1][0]

        task_output = ''.join(cls._get_dialogue_labels(
            labels, labels_metadata, augmentation, label_type, interaction, chunk=chunk, response=response
        ))
        # Compose output
        output = f'{task_starter}\n\n{task_output}'.strip()
        output = output + eos

        return output

    @classmethod
    def _compose_data_discriminator(
            cls,
            tokenizer: PreTrainedTokenizer,
            len_prefix_tok: int,
            len_suffix_tok: Union[int, List[int]],
            augmentation: bool,
            label_type: Literal['global', 'local'],
            interaction: str,
            dialogue_lines: Optional[List[Tuple[str, str, str]]] = None,
            tgt_lines: Optional[List[str]] = None,
            chunk: Optional[str] = None,
            max_chunk_turns: Optional[int] = None,
            context: Optional[str] = None,
            response: Optional[str] = None,
            max_context_turns: Optional[int] = None
    ) -> List[str]:
        # Depending on the labelling type compose the strings
        if label_type == 'global':
            # Additional parameters
            if augmentation:
                chunk_intro = random.choice(cls.DIALOGUE_CHUNK_SEP).replace(DIALOGUE_CHUNK, chunk)
                chunk_intro = chunk_intro.replace(INTERACTION, interaction)
            else:
                chunk_intro = cls.DIALOGUE_CHUNK_SEP[0].replace(DIALOGUE_CHUNK, chunk)
            # Compose input
            input_list = cls._compose_data_global_label(
                tokenizer, len_prefix_tok, len_suffix_tok, chunk_intro, dialogue_lines, max_chunk_turns
            )
        elif label_type == 'local':
            # Additional parameters
            if augmentation:
                context_intro = random.choice(cls.DIALOGUE_CONTEXT_SEP).replace(CONTEXT, context)
                context_intro = context_intro.replace(INTERACTION, interaction)
                response_intro = random.choice(cls.DIALOGUE_RESPONSE_SEP).replace(RESPONSE, response)
            else:
                context_intro = cls.DIALOGUE_CONTEXT_SEP[0].replace(CONTEXT, context)
                response_intro = cls.DIALOGUE_RESPONSE_SEP[0].replace(RESPONSE, response)
            # Compose input
            input_list = cls._compose_data_local_label(
                tokenizer,
                len_prefix_tok,
                len_suffix_tok,
                response_intro=response_intro,
                response_lines=tgt_lines,
                context_intro=context_intro,
                context_lines=dialogue_lines,
                max_context_turns=max_context_turns
            )
        else:
            raise ValueError(f'Unknown label type: {label_type}. Accepted model types are \'global\' and \'local\'.')

        return input_list

    @classmethod
    def _compose_discriminator_dialogue(
            cls,
            label_type: Literal['global', 'local'],
            tokenizer: PreTrainedTokenizer,
            speakers: Tuple[str, str],
            dialogue_lines: List[str],
            interaction: str,
            augmentation: bool,
            dropout: bool,
            bot: bool,
            chunk: Optional[str] = None,
            context: Optional[str] = None,
            response: Optional[str] = None,
            global_labels_metadata: Optional[Dict] = None,
            global_labels: Optional[Dict[str, str]] = None,
            tgt_global_labels_metadata: Optional[Dict] = None,
            tgt_global_labels: Optional[Dict[str, str]] = None,
            line_labels_metadata: Optional[Dict] = None,
            line_labels: Optional[List[Dict[str, str]]] = None,
            tgt_line_labels_metadata: Optional[Dict] = None,
            tgt_line_labels: Optional[Dict[str, str]] = None,
            max_chunk_turns: Optional[int] = None,
            max_context_turns: Optional[int] = None
    ) -> List[str]:
        assert not (
                tgt_global_labels_metadata is not None and len(tgt_global_labels) > 0 and
                tgt_line_labels_metadata is not None and len(tgt_line_labels) > 0
        )
        assert (label_type != 'global') or (tgt_global_labels_metadata is not None and len(tgt_global_labels) > 0)
        assert (label_type != 'local') or (tgt_line_labels_metadata is not None and len(tgt_line_labels) > 0)
        # Get additional parameters depending on the type of label
        sep = random.choices(cls.SEP_SYM, weights=cls.SEP_SYM_P)[0] if augmentation else cls.SEP_SYM[0]
        if label_type == 'global':
            # Task instance parameters
            tgt_labels_metadata = tgt_global_labels_metadata
            tgt_labels = tgt_global_labels
            # Data starter
            starter = cls._compose_dialogue_starter(
                'discriminator', augmentation, interaction, global_labels=global_labels, global_labels_metadata=global_labels_metadata
            )
            # Classification object
            tgt_lines = None
            # Classification target output
            targets = cls._compose_discriminator_task(
                tgt_labels, tgt_labels_metadata, augmentation, label_type, tokenizer.eos_token, interaction, chunk=chunk
            )
            len_suffix_tok = len(tokenizer(targets).input_ids)
            targets = repeat(targets)
        elif label_type == 'local':
            # Task instance parameters
            tgt_labels_metadata = tgt_line_labels_metadata
            tgt_labels = tgt_line_labels
            # Data starter
            starter = cls._compose_dialogue_starter(
                'discriminator', augmentation, interaction, global_labels=global_labels, global_labels_metadata=global_labels_metadata
            )
            # Classification object
            tgt_lines = cls._get_dialogue_lines(
                dialogue_lines,
                speakers,
                augmentation,
                dropout,
                sep=sep
                # labels=tgt_line_labels,
                # line_labels_metadata=tgt_line_labels_metadata
            )
            # Classification target output
            targets = [
                cls._compose_discriminator_task(
                    tgt_labels_, tgt_labels_metadata, augmentation, label_type, tokenizer.eos_token, interaction, response=response
                ) for tgt_labels_ in tgt_labels
            ]
            len_suffix_tok = [len(ids) for ids in tokenizer(targets).input_ids]
        else:
            raise ValueError('Missing label data')
        # Task description
        task_description = cls._compose_task_description(
            speakers,
            'discriminator',
            interaction,
            augmentation,
            dropout,
            bot,
            global_labels_metadata=global_labels_metadata,
            line_labels_metadata=line_labels_metadata,
            tgt_labels_metadata=tgt_labels_metadata,
            label_type=label_type,
            chunk=chunk,
            context=context,
            response=response
        )
        len_prefix_tok = len(tokenizer(f'{task_description}{starter}').input_ids)
        # Dialogue context
        dialogue_lines = cls._get_dialogue_lines(
            dialogue_lines,
            speakers,
            augmentation,
            dropout,
            labels=line_labels,
            line_labels_metadata=line_labels_metadata,
            split_lines=True,
            sep=sep
        )
        # Get dialogue data
        dialogue_data = cls._compose_data_discriminator(
            tokenizer,
            len_prefix_tok,
            len_suffix_tok,
            augmentation,
            label_type,
            dialogue_lines=dialogue_lines,
            tgt_lines=tgt_lines,
            chunk=chunk,
            max_chunk_turns=max_chunk_turns,
            interaction=interaction,
            context=context,
            response=response,
            max_context_turns=max_context_turns
        )

        # Compose discriminator model input output
        seqs = [
            f'{task_description}{starter}{dialogue_passage}{target}'.strip()
            for dialogue_passage, target in zip(dialogue_data, targets)
        ]

        return seqs

    @classmethod
    def _compose_discriminator_dialogue_eval(
            cls,
            label_type: Literal['global', 'local'],
            label: str,
            approach: Literal['posterior', 'infilling', 'prediction'],
            tokenizer: PreTrainedTokenizer,
            speakers: Tuple[str, str],
            dialogue_lines: List[str],
            interaction: str,
            chunk: Optional[str] = None,
            context: Optional[str] = None,
            response: Optional[str] = None,
            global_labels_metadata: Optional[Dict] = None,
            global_labels: Optional[Dict[str, str]] = None,
            line_labels_metadata: Optional[Dict] = None,
            line_labels: Optional[List[Dict[str, str]]] = None,
            max_chunk_turns: Optional[int] = None,
            max_context_turns: Optional[int] = None
    ):
        if label_type == 'global':
            #
            tmp_global_labels = {label: LABEL_VALUE}
            #
            if approach == 'posterior':
                # Task description
                task_description = cls._compose_task_description(
                    speakers,
                    'discriminator',
                    interaction,
                    False,
                    False,
                    False,
                    global_labels_metadata=global_labels_metadata,
                    tgt_labels_metadata=global_labels_metadata,
                    label_type=label_type,
                    chunk=chunk
                )
                len_prefix_tok = len(tokenizer(f'{task_description}').input_ids)
                # Classification target output
                target = cls._compose_discriminator_task(
                    tmp_global_labels, global_labels_metadata, False, label_type, tokenizer.eos_token, interaction, chunk=chunk
                )
                len_suffix_tok = max(
                    len(ids) for ids in tokenizer(
                        [target.replace(LABEL_VALUE, value) for value in global_labels_metadata[label]['values']]
                    ).input_ids
                )
                # Dialogue context
                dialogue_lines = cls._get_dialogue_lines(
                    dialogue_lines,
                    speakers,
                    False,
                    False,
                    split_lines=True
                )
                # Get dialogue data
                dialogue_data = cls._compose_data_discriminator(
                    tokenizer,
                    len_prefix_tok,
                    len_suffix_tok,
                    False,
                    label_type,
                    dialogue_lines=dialogue_lines,
                    chunk=chunk,
                    max_chunk_turns=max_chunk_turns,
                    interaction=interaction
                )
                # Compose discriminator model input output
                seqs = {
                    'prompt': f'{task_description}{dialogue_data[0]}',
                    'annotations': [
                        target.replace(LABEL_VALUE, value)
                        for value in global_labels_metadata[label]['values']
                    ],
                    'target': global_labels[label]
                }
            elif approach == 'infilling' or approach == 'prediction':
                # Additional parameters
                task_description = cls._compose_task_description(
                    speakers,
                    'generator',
                    interaction,
                    False,
                    False,
                    False,
                    global_labels_metadata=global_labels_metadata,
                    tgt_labels_metadata=global_labels_metadata
                )
                starter = cls._compose_dialogue_starter(
                    'generator', False, interaction, global_labels=tmp_global_labels, global_labels_metadata=global_labels_metadata
                )
                dialogue_lines = cls._get_dialogue_lines(
                    dialogue_lines,
                    speakers,
                    False,
                    False,
                    eos=tokenizer.eos_token
                )
                # Compose generator model input output (i.e., the dialogue) for discrimination
                seqs = {
                    'task_description': f'{task_description}',
                    'starter': [starter.replace(LABEL_VALUE, value) for value in global_labels_metadata[label]['values']],
                    'utterances': dialogue_lines
                }
            else:
                raise ValueError(f'Unknown approach requested \'{approach}\'')
        elif label_type == 'local':
            # Create dummy target labels
            tmp_line_labels = [{label: LABEL_VALUE}] * len(line_labels)
            #
            if approach == 'posterior':
                # Task description
                task_description = cls._compose_task_description(
                    speakers,
                    'discriminator',
                    interaction,
                    False,
                    False,
                    False,
                    line_labels_metadata=line_labels_metadata,
                    tgt_labels_metadata=line_labels_metadata,
                    label_type=label_type,
                    context=context,
                    response=response
                )
                len_prefix_tok = len(tokenizer(f'{task_description}').input_ids)
                # Classification target output
                targets = [
                    cls._compose_discriminator_task(
                        tmp_line_labels_, line_labels_metadata, False, label_type, tokenizer.eos_token, interaction, response=response
                    ) for tmp_line_labels_ in tmp_line_labels
                ]
                len_suffix_tok = [
                    max(
                        len(ids) for ids in tokenizer(
                            [target.replace(LABEL_VALUE, value) for value in line_labels_metadata[label]['values']]
                        ).input_ids
                    ) for target in targets
                ]
                # Classification object
                tgt_lines = cls._get_dialogue_lines(
                    dialogue_lines,
                    speakers,
                    False,
                    False
                )
                # Dialogue context
                dialogue_lines = cls._get_dialogue_lines(
                    dialogue_lines,
                    speakers,
                    False,
                    False,
                    labels=line_labels,
                    line_labels_metadata=line_labels_metadata,
                    split_lines=True
                )
                # Get dialogue data
                dialogue_data = cls._compose_data_discriminator(
                    tokenizer,
                    len_prefix_tok,
                    len_suffix_tok,
                    False,
                    label_type,
                    dialogue_lines=dialogue_lines,
                    tgt_lines=tgt_lines,
                    interaction=interaction,
                    context=context,
                    response=response,
                    max_context_turns=max_context_turns
                )
                # Compose discriminator model input output
                seqs = {
                    'task_description': f'{task_description}',
                    'passages': [
                        {
                            'context_response': dialogue_passage,
                            'annotations': [
                                annotation.replace(LABEL_VALUE, value)
                                for value in line_labels_metadata[label]['values']
                            ],
                            'target': target[label]
                        }
                        for dialogue_passage, annotation, target in zip(dialogue_data, targets, line_labels)
                    ]
                }
            elif approach == 'infilling' or approach == 'prediction':
                # Additional parameters
                task_description = cls._compose_task_description(
                    speakers,
                    'generator',
                    interaction,
                    False,
                    False,
                    False,
                    line_labels_metadata=line_labels_metadata,
                    tgt_labels_metadata=line_labels_metadata
                )
                starter = cls._compose_dialogue_starter('generator', False, interaction)  # , global_labels=global_labels)
                dialogue_lines = cls._get_dialogue_lines(
                    dialogue_lines,
                    speakers,
                    False,
                    False,
                    labels=tmp_line_labels,
                    line_labels_metadata=line_labels_metadata,
                    split_lines=True,
                    eos=tokenizer.eos_token
                )
                # Compose generator model input output (i.e., the dialogue) for discrimination
                seqs = {
                    'task_description': f'{task_description}{starter}',
                    'utterances': [
                        {
                            'prompts': [
                                prompt.replace(LABEL_VALUE, value)
                                for value in line_labels_metadata[label]['values']
                            ],
                            'target': target[label],
                            'text': text[len(prompt):]
                        }
                        for (_, prompt, text), target in zip(dialogue_lines, line_labels)
                    ]
                }
            else:
                raise ValueError(f'Unknown approach requested \'{approach}\'')
        else:
            raise ValueError(f'Unknown label type \'{type}\'')

        return seqs

    @classmethod
    def _compose_generator_dialogue(
            cls,
            tokenizer: PreTrainedTokenizer,
            speakers: Tuple[str, str],
            dialogue_lines: List[str],
            interaction: str,
            augmentation: bool,
            dropout: bool,
            bot: bool,
            global_labels_metadata: Optional[Dict] = None,
            global_labels: Optional[Dict[str, str]] = None,
            line_labels_metadata: Optional[Dict] = None,
            line_labels: Optional[List[Dict[str, str]]] = None,
            min_turns: int = 1
    ) -> List[str]:
        # Additional parameters
        task_description = cls._compose_task_description(
            speakers,
            'generator',
            interaction,
            augmentation,
            dropout,
            bot,
            global_labels_metadata=global_labels_metadata,
            line_labels_metadata=line_labels_metadata
        )
        starter = cls._compose_dialogue_starter(
            'generator', augmentation, interaction, global_labels=global_labels, global_labels_metadata=global_labels_metadata
        )
        dialogue_lines = cls._get_dialogue_lines(
            dialogue_lines,
            speakers,
            augmentation,
            dropout,
            labels=line_labels,
            line_labels_metadata=line_labels_metadata,
            eos=tokenizer.eos_token
        )
        dialogue_splits = cls._compose_data_dialogue(
            tokenizer, dialogue_lines, len(tokenizer(f'{task_description}{starter}').input_ids), min_turns
        )
        # Compose generator model input output (i.e., the dialogue)
        seqs = [f'{task_description}{starter}{dialogue_split}'.strip() for dialogue_split in dialogue_splits]

        return seqs

    @classmethod
    def _compose_generator_dialogue_eval(
            cls,
            speakers: Tuple[str, str],
            dialogue_lines: List[str],
            interaction: str,
            global_labels_metadata: Optional[Dict] = None,
            global_labels: Optional[Dict[str, str]] = None,
            line_labels_metadata: Optional[Dict] = None,
            line_labels: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Union[str, List[Tuple[str, str]]]]:
        # Additional parameters
        task_description = cls._compose_task_description(
            speakers,
            'generator',
            interaction,
            False,
            False,
            False,
            global_labels_metadata=global_labels_metadata,
            line_labels_metadata=line_labels_metadata,
        )
        starter = cls._compose_dialogue_starter(
            'generator', False, interaction, global_labels=global_labels, global_labels_metadata=global_labels_metadata)
        dialogue_lines = cls._get_dialogue_lines(
            dialogue_lines,
            speakers,
            False,
            False,
            labels=line_labels,
            line_labels_metadata=line_labels_metadata,
            split_lines=True
        )
        # Compose generator model input output (i.e., the dialogue)
        dialogue = {
            'task_description': f'{task_description}{starter}',
            'utterances': [(prompt, text[len(prompt):]) for _, prompt, text in dialogue_lines]
        }

        return dialogue
