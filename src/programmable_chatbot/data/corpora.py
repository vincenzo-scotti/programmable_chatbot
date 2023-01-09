import os

import math
import random
import re
from copy import deepcopy

from joblib import Parallel
from joblib import delayed
from joblib import parallel_backend

import json
import pandas as pd
from sklearn.model_selection import train_test_split

from .utils import _DialogueCorpus
from .utils import (
    PLACEHOLDER_F_STRING,
    SPEAKERS,
    INTERACTION,
    SPEAKER_DESCRIPTIONS,
    RESPONSE,
    LABEL_ID,
    DIALOGUE_CHUNK,
    CLS_SEP,
    NULL
)

from transformers import PreTrainedTokenizer

from typing import Optional, Union, Tuple, List, Dict, Pattern, Literal


EMO_SPEAKER: str = PLACEHOLDER_F_STRING.format('emotionalspeaker')
EMP_LISTENER: str = PLACEHOLDER_F_STRING.format('empatheticlistener')
PERSONA_1: str = PLACEHOLDER_F_STRING.format('persona1')
PERSONA_2: str = PLACEHOLDER_F_STRING.format('persona2')
THERAPIST: str = PLACEHOLDER_F_STRING.format('therapistspeaker')
PATIENT: str = PLACEHOLDER_F_STRING.format('patientspeaker')


# Open domain data sets

class DailyDialog(_DialogueCorpus):
    # Corpus metadata
    IDENTIFIER = 'dailydialog'
    GLOBAL_LABELS_METADATA: Dict[str, Dict[str, Optional[str]]] = {
        'topic': {
            'id': 'topic',
            'description': 'The topic is the argument of the interaction',
            'values': [
                'work', 'relationship', 'school life', 'politics', 'health', 'finance', 'ordinary life',
                'attitude and emotion', 'tourism', 'culture and education'
            ]
        }
    }
    LINE_LABELS_METADATA: Dict[str, Dict[str, Optional[str]]] = {
        'emotion': {
            'id': 'emotion',
            'description': 'The emotion is the categorical affective state characterising '
                           'a speakers during the interaction. '
                           'The emotion can be recognised by what the speaker is saying',
            'values': ['neutral', 'disgust', 'surprise', 'anger', 'happiness', 'sadness', 'fear']
        },
        'dialogue_act': {
            'id': 'dialogue act',
            'description': 'Dialog acts represent the communication functions '
                           'when people say something in a conversation. '
                           'Dialogue acts are used to indicate the communication intention',
            'values': ['inform', 'question', 'directive', 'commissive']
        }
    }
    #
    TOPIC_DECODER: Dict[str, str] = {
        'work': 'work',
        'relationship': 'relationship',
        'school_life': 'school life',
        'politics': 'politics',
        'health': 'health',
        'finance': 'finance',
        'ordinary_life': 'ordinary life',
        'attitude_and_emotion': 'attitude and emotion',
        'tourism': 'tourism',
        'culture_and_education': 'culture and education'
    }
    EMOTION_DECODER: Dict[str, str] = {
        'no_emotion': 'neutral',
        'disgust': 'disgust',
        'surprise': 'surprise',
        'anger': 'anger',
        'happiness': 'happiness',
        'sadness': 'sadness',
        'fear': 'fear'
    }

    def _preprocess_dialogue(self, dialogue: Dict) -> Dict[str, Union[str, Dict[str, str]]]:
        # Correct misspelled topic label
        if dialogue['topic'] == 'culture_and_educastion':
            dialogue['topic'] = 'culture_and_education'
        # Pre-processed dialogue
        dialogue: Dict[str, Union[str, Dict[str, str]]] = {
            'split': self.split,
            'corpus': 'DailyDialog',
            'topic': self.TOPIC_DECODER[dialogue['topic']],
            'utterances': [
                {
                    'emotion': self.EMOTION_DECODER[utterance['emotion']],
                    'dialogue_act': utterance['act'],
                    'text': self._preprocess_text(utterance['text'])
                } for utterance in dialogue['dialogue']
            ]
        }

        return dialogue

    def _load_samples(self) -> List[Dict[str, Union[str, Dict[str, str]]]]:
        # Get split file name
        if self.split == 'train':
            file_name = 'train.json'
        elif self.split == 'validation':
            file_name = 'valid.json'
        elif self.split == 'test':
            file_name = 'test.json'
        else:
            raise ValueError(f'Unknown value for data set split: {self.split}')
        # Load split of the corpus
        with open(os.path.join(self.corpus_dir_path, file_name)) as f:
            dialogues = [json.loads(line) for line in f]
        # Apply subsampling if required
        if self.sample is not None:
            # Get number of samples to collect
            n_samples = self.sample if isinstance(self.sample, int) else int(math.ceil(self.sample * len(dialogues)))
            # Subsample data set unless the number of samples to take is equal to the number of samples available
            if n_samples != len(dialogues):
                dialogues = random.sample(dialogues, n_samples)
        # Standardise corpus
        with parallel_backend(self.joblib_backend, n_jobs=self.n_jobs):
            return Parallel(verbose=self.verbosity_level)(
                delayed(self._preprocess_dialogue)(dialogue) for dialogue in dialogues
            )

    @classmethod
    def _get_labels(
            cls,
            dialogue: Dict,
            augmentation: bool,
            global_labels_metadata: Optional[Dict[str, str]],
            line_labels_metadata: Optional[Dict[str, str]]
    ) -> Tuple[Optional[Dict[str, str]], Optional[List[Dict[str, str]]]]:
        # Global labels
        global_labels = {'topic': dialogue['topic']}
        # Local labels
        local_labels = [
            {'emotion': utterance['emotion'], 'dialogue_act': utterance['dialogue_act']}
            for utterance in dialogue['utterances']
        ]
        return global_labels, local_labels


class EmpatheticDialogues(_DialogueCorpus):
    # Corpus metadata
    IDENTIFIER = 'empatheticdialogues'
    GLOBAL_LABELS_METADATA: Dict[str, Dict[str, Optional[str]]] = {
        'context': {
            'id': f'{EMO_SPEAKER} emotion',
            'description': f'The emotion of the {EMO_SPEAKER} serves as context for emotional situation grounding. '
                           f'It describes the affective state of the {EMO_SPEAKER} and can be recognised by '
                           f'what the {EMO_SPEAKER} is saying'
        }
    }
    GENERATIVE_TASK_PREFIX: List[str] = [
        f'In the following interactions, {SPEAKERS} will converse in natural language. '
        f'The {EMO_SPEAKER} describes a situation and the {EMP_LISTENER} reacts empathetically.',
        f'During these interactions, {SPEAKERS} will converse in natural language. '
        f'The {EMO_SPEAKER} tells a story and the {EMP_LISTENER} reacts empathetically.',
        f'In the following interactions, {SPEAKERS} converse in natural language. '
        f'The {EMO_SPEAKER} describes a situation and the {EMP_LISTENER} reacts empathetically.',
        f'During these interactions, {SPEAKERS} converse in natural language. '
        f'The {EMO_SPEAKER} tells a story and the {EMP_LISTENER} reacts empathetically.',
        f'In the following interactions, {SPEAKERS} are conversing in natural language. '
        f'The {EMO_SPEAKER} describes a situation and the {EMP_LISTENER} reacts empathetically.',
        f'During these interactions, {SPEAKERS} are conversing in natural language. '
        f'The {EMO_SPEAKER} tells a story and the {EMP_LISTENER} reacts empathetically.'
    ]
    EMO_HUMAN_PARTICIPANT: List[str] = ['an emotional human speaker', 'a human speaker', 'a human', 'a person', 'a human user']
    EMP_HUMAN_PARTICIPANT: List[str] = ['an empathetic human listener', 'a human listener', 'a human', 'a person', 'a human user']
    EMO_BOT_PARTICIPANT: List[str] = [
        'an emotional AI assistant', 'an AI assistant', 'an emotional AI system', 'an AI system', 'an emotional AI', 'an AI',
        'an emotional chatbot', 'a chatbot', 'an emotional conversational agent', 'a conversational agent',
        'an emotional dialogue agent', 'a dialogue agent'
    ]
    EMP_BOT_PARTICIPANT: List[str] = [
        'an empathetic AI assistant', 'an AI assistant', 'an empathetic AI system', 'an AI system', 'an empathetic AI', 'an AI',
        'an empathetic chatbot', 'a chatbot', 'an empathetic conversational agent', 'a conversational agent',
        'an empathetic dialogue agent', 'a dialogue agent'
    ]
    DISCRIMINATIVE_TASK_PROMPT: Dict[str, List[str]] = {
        'global': [
            f'{LABEL_ID}{CLS_SEP}',
            f'{LABEL_ID} in the {INTERACTION} {DIALOGUE_CHUNK} {CLS_SEP}',
            f'The {LABEL_ID} in the {INTERACTION} {DIALOGUE_CHUNK} is'
        ],
        'local': [f'{LABEL_ID}{CLS_SEP}', f'{RESPONSE} {LABEL_ID}{CLS_SEP}', f'The {LABEL_ID} of the {RESPONSE} is']
    }
    LBL_DESC_SYM: List[Tuple[str, str]] = [('', '. ')]
    LBL_BOL_EOL_SYM: List[Tuple[str, str]] = [('', '.\n'), ('', '. ')]
    LBL_DESC_SYM_P: List[float] = [1.0]
    LBL_BOL_EOL_SYM_P: List[float] = [0.7, 0.3]
    # IDs
    HUMAN_SPEAKER_IDS: Dict[str, List[Tuple[str, str]]] = {
        'default': [
            ('Emotional Speaker', 'Empathetic Listener'),
            ('Emotional User', 'Empathetic Listener'),
            ('Emotional User', 'Empathetic User')
        ],
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
            ('AI', 'Person'),
            ('AI', 'User'),
            ('Emotional AI', 'Person'),
            ('Emotional AI', 'User'),
            ('Person', 'AI'),
            ('User', 'AI'),
            ('Person', 'Empathetic AI'),
            ('User', 'Empathetic AI')
        ],
        'chatbot': [
            ('Chatbot', 'Person'),
            ('Chatbot', 'User'),
            ('Emotional Chatbot', 'Person'),
            ('Emotional Chatbot', 'User'),
            ('Person', 'Chatbot'),
            ('User', 'Chatbot'),
            ('Person', 'Empathetic Chatbot'),
            ('User', 'Empathetic Chatbot')
        ]
    }
    # Data loading parameters
    DF_COLS = [
        'conv_id', 'utterance_idx', 'context', 'prompt', 'speaker_idx', 'utterance', 'selfeval', 'tags', 'distractors'
    ]
    D_TYPES = [str, int, str, str, int, str, str, str, str]

    def _preprocess_text(self, text: str) -> str:
        text = text.replace('_comma_', ',')
        return super(EmpatheticDialogues, self)._preprocess_text(text)

    def _preprocess_dialogue(self, dialogue: pd.DataFrame) -> Dict[str, Union[str, Dict[str, str]]]:
        # Pre-processed dialogue
        dialogue: Dict[str, Union[str, Dict[str, str]]] = {
            'split': self.split,
            'corpus': 'EmpatheticDialogues',
            'context': dialogue['context'].unique().item(),
            'utterances': [
                {'text': self._preprocess_text(utterance['utterance'])}
                for _, utterance in dialogue.iterrows()
            ]
        }

        return dialogue

    def _load_samples(self) -> List[Dict[str, Union[str, Dict[str, str]]]]:
        # Get split file name
        if self.split == 'train':
            file_name = 'train.csv'
        elif self.split == 'validation':
            file_name = 'valid.csv'
        elif self.split == 'test':
            file_name = 'test.csv'
        else:
            raise ValueError(f'Unknown value for data set split: {self.split}')
        # Load split of the corpus
        with open(os.path.join(self.corpus_dir_path, file_name)) as f:
            dialogues = [line.strip().split(',') for line in f]
        dialogues.pop(0)
        dialogues = [sample if len(sample) == len(self.DF_COLS) else sample + [None] for sample in dialogues]
        dialogues = [[t(x) if x is not None else x for x, t in zip(sample, self.D_TYPES)] for sample in dialogues]
        dialogues = pd.DataFrame(data=dialogues, columns=self.DF_COLS)
        dialogues = [dialogue for _, dialogue in dialogues.groupby('conv_id')]
        # Apply subsampling if required
        if self.sample is not None:
            # Get number of samples to collect
            n_samples = self.sample if isinstance(self.sample, int) else int(math.ceil(self.sample * len(dialogues)))
            # Subsample data set unless the number of samples to take is equal to the number of samples available
            if n_samples != len(dialogues):
                dialogues = random.sample(dialogues, n_samples)
        # Standardise corpus
        with parallel_backend(self.joblib_backend, n_jobs=self.n_jobs):
            return Parallel(verbose=self.verbosity_level)(
                delayed(self._preprocess_dialogue)(dialogue) for dialogue in dialogues
            )

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
            speakers = cls.HUMAN_SPEAKER_IDS['default'][0]
            bot = False

        return speakers, bot

    @classmethod
    def _get_labels_metadata(
            cls, speakers: Tuple[str, str], dialogue: Dict, augmentation: bool, dropout: bool
    ) -> Tuple[Dict[str, str], Dict[str, str]]:
        global_labels_metadata, line_labels_metadata = super(EmpatheticDialogues, cls)._get_labels_metadata(
            speakers, dialogue, augmentation, dropout
        )
        global_labels_metadata = deepcopy(global_labels_metadata)
        global_labels_metadata['context']['id'] = global_labels_metadata['context']['id'].replace(EMO_SPEAKER, speakers[0])

        return global_labels_metadata, line_labels_metadata

    @classmethod
    def _get_labels(
            cls,
            dialogue: Dict,
            augmentation: bool,
            global_labels_metadata: Optional[Dict[str, str]],
            line_labels_metadata: Optional[Dict[str, str]]
    ) -> Tuple[Optional[Dict[str, str]], Optional[List[Dict[str, str]]]]:
        global_labels = {'context': dialogue['context']} if global_labels_metadata is not None and len(global_labels_metadata) > 0 else None
        line_labels = None

        return global_labels, line_labels

    @classmethod
    def _compose_premise(cls, speakers: Tuple[str, str], interaction: str, augmentation: bool, bot: bool) -> str:
        # Presentation line
        presentation = random.choice(cls.PRESENTATION) if augmentation else cls.PRESENTATION[0]
        # Generate speakers introduction and description
        sp1, sp2 = speakers
        if bot:
            if any(sp1.lower() in key.lower() or key.lower() in sp1.lower() for key in cls.HUMAN_BOT_SPEAKER_IDS):
                p1_list, p2_list = cls.EMO_BOT_PARTICIPANT, cls.EMP_HUMAN_PARTICIPANT
            else:
                p1_list, p2_list = cls.EMO_HUMAN_PARTICIPANT, cls.EMP_BOT_PARTICIPANT
        else:
            p1_list, p2_list = cls.EMO_HUMAN_PARTICIPANT, cls.EMP_HUMAN_PARTICIPANT
        p1 = random.choice(p1_list) if augmentation else p1_list[0]
        p2 = random.choice(p2_list) if augmentation else p2_list[0]
        speakers_description = f'{p1}, {presentation} {sp1}, and {p2}, {presentation} {sp2}'
        # Compose premise
        premise = random.choice(cls.PREMISE) if augmentation else cls.PREMISE[0]
        premise = premise.replace(INTERACTION, interaction)
        premise = premise.replace(SPEAKER_DESCRIPTIONS, speakers_description)
        premise = premise.replace(EMO_SPEAKER, sp1)
        premise = premise.replace(EMP_LISTENER, sp2)

        return premise

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
        sp1, sp2 = speakers
        task_description = super(EmpatheticDialogues, cls)._compose_task_description(
            speakers, model_type, interaction, augmentation, dropout, bot,
            global_labels_metadata=global_labels_metadata, line_labels_metadata=line_labels_metadata,
            tgt_labels_metadata=tgt_labels_metadata, label_type=label_type, chunk=chunk, context=context, response=response
        )
        task_description = task_description.replace(EMO_SPEAKER, sp1)
        task_description = task_description.replace(EMP_LISTENER, sp2)

        return task_description


class PersonaChat(_DialogueCorpus):
    # Corpus metadata
    IDENTIFIER = 'personachat'
    GLOBAL_LABELS_METADATA: Dict[str, Dict[str, Optional[str]]] = {
        'persona': {
            'id': 'persona description of the speakers',
            'description': 'A persona description is a short description in a few sentences of the personal '
                           'information of one or both speakers'
        }
    }
    P_DROP_PERSONA: float = 0.33
    LBL_DESC_SYM: List[Tuple[str, str]] = [('', '. ')]
    LBL_BOL_EOL_SYM: List[Tuple[str, str]] = [('', '')]
    LBL_DESC_SYM_P: List[float] = [1.0]
    LBL_BOL_EOL_SYM_P: List[float] = [1.0]
    # Data loading parameters
    DISTRACTORS_SPLIT_SYM = '\t\t'
    DISTRACTORS_SEPARATOR_SYM = '|'
    UTTERANCE_SEPARATOR_SYM = '\t'
    Y_PERSONA_SYM = 'your persona:'
    P_PERSONA_SYM = 'partner\'s persona:'

    def _preprocess_dialogue(self, data: List[str]) -> Dict[str, Union[str, Dict[str, str]]]:
        # Self persona (first speaker persona)
        self_persona: List[str] = [
            elem.split(self.Y_PERSONA_SYM)[-1].strip() for elem in data if elem.startswith(self.Y_PERSONA_SYM)
        ]
        # Other persona (second speaker persona)
        other_persona: List[str] = [
            elem.split(self.P_PERSONA_SYM)[-1].strip() for elem in data if elem.startswith(self.P_PERSONA_SYM)
        ]
        # Dialogue
        dialogue: List[str] = [
            utterance for elem in data
            for utterance in elem.split(self.DISTRACTORS_SPLIT_SYM)[0].split(self.UTTERANCE_SEPARATOR_SYM)
            if not (elem.startswith(self.Y_PERSONA_SYM) or elem.startswith(self.P_PERSONA_SYM))
        ]
        # Pre-processed dialogue
        dialogue: Dict[str, Union[str, Dict[str, str]]] = {
            'split': self.split,
            'corpus': 'Persona-Chat',
            'self_persona': self._preprocess_text(' '.join(self_persona)),
            'other_persona': self._preprocess_text(' '.join(other_persona)),
            'utterances': [{'text': self._preprocess_text(utterance)} for utterance in dialogue]
        }

        return dialogue

    def _load_samples(self) -> List[Dict[str, Union[str, Dict[str, str]]]]:
        # Get split file name
        if self.split == 'train':
            file_name = 'train_both_original.txt'
        elif self.split == 'validation':
            file_name = 'valid_both_original.txt'
        elif self.split == 'test':
            file_name = 'test_both_original.txt'
        else:
            raise ValueError(f'Unknown value for data set split: {self.split}')
        # Load split of the corpus
        with open(os.path.join(self.corpus_dir_path, file_name)) as f:
            dialogues = [line.split(' ', 1) for line in f.readlines()]
        idxs = [line_idx for line_idx, (i, _) in enumerate(dialogues) if int(i) == 1]
        idxs = [*zip(idxs, idxs[1:] + [len(dialogues)])]
        dialogues = [[elem for _, elem in dialogues[s_idx:e_idx]] for s_idx, e_idx in idxs]
        # Apply subsampling if required
        if self.sample is not None:
            # Get number of samples to collect
            n_samples = self.sample if isinstance(self.sample, int) else int(
                math.ceil(self.sample * len(dialogues)))
            # Subsample data set unless the number of samples to take is equal to the number of samples available
            if n_samples != len(dialogues):
                dialogues = random.sample(dialogues, n_samples)
        # Standardise corpus
        with parallel_backend(self.joblib_backend, n_jobs=self.n_jobs):
            return Parallel(verbose=self.verbosity_level)(
                delayed(self._preprocess_dialogue)(dialogue) for dialogue in dialogues
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
        return super(PersonaChat, self).get_data_for_fitting(
            full=full, plaintext=plaintext, dropout=dropout, augmentation=augmentation, generator=generator,
            discriminator=False
        )

    @classmethod
    def _get_labels(
            cls,
            dialogue: Dict,
            augmentation: bool,
            global_labels_metadata: Optional[Dict[str, str]],
            line_labels_metadata: Optional[Dict[str, str]]
    ) -> Tuple[Optional[Dict[str, str]], Optional[List[Dict[str, str]]]]:
        persona_descriptions = [
            f'{PERSONA_1} persona: {dialogue["self_persona"]}', f'{PERSONA_2} persona: {dialogue["other_persona"]}'
        ]
        if augmentation:
            persona_descriptions = random.sample(persona_descriptions, len(persona_descriptions))
        if augmentation and random.uniform(0.0, 1.0) < cls.P_DROP_PERSONA:
            persona_descriptions = random.choice(persona_descriptions)
        else:
            persona_descriptions = '\n'.join(persona_descriptions)

        global_labels = {'persona': '\n' + persona_descriptions} if global_labels_metadata is not None else None
        line_labels = None

        return global_labels, line_labels

    @classmethod
    def _compose_dialogue_starter(
            cls,
            speakers: Tuple[str, str],
            *args,
            **kwargs
    ) -> str:
        sp1, sp2 = speakers
        starter = super(PersonaChat, cls)._compose_dialogue_starter(speakers, *args, **kwargs)
        starter = starter.replace(PERSONA_1, sp1)
        starter = starter.replace(PERSONA_2, sp2)

        if starter.endswith('..'):
            starter = starter[:-1]

        return starter


class WizardOfWikipedia(_DialogueCorpus):
    # Corpus metadata
    IDENTIFIER = 'wizard_of_wikipedia'
    GLOBAL_LABELS_METADATA: Dict[str, Dict[str, Optional[str]]] = {
        'topic': {'id': 'topic', 'description': 'The topic is the argument of the interaction'}
    }
    LBL_DESC_SYM: List[Tuple[str, str]] = [('', '. ')]
    LBL_BOL_EOL_SYM: List[Tuple[str, str]] = [('', '.\n'), ('', '. ')]
    LBL_DESC_SYM_P: List[float] = [1.0]
    LBL_BOL_EOL_SYM_P: List[float] = [0.7, 0.3]

    @staticmethod
    def _convert_speaker_id(speaker) -> Literal['wizard', 'apprentice']:
        if 'wizard' in speaker.lower():
            return 'wizard'
        elif 'apprentice' in speaker.lower():
            return 'apprentice'
        else:
            raise ValueError(f'Unknown speaker: {speaker}')

    def _preprocess_dialogue(self, dialogue: Dict) -> Dict[str, Union[str, Dict[str, str]]]:
        # Pre-processed dialogue
        dialogue: Dict[str, Union[str, Dict[str, str]]] = {
            'split': self.split,
            'corpus': 'Wizard of Wikipedia',
            'topic': dialogue['chosen_topic'],
            'utterances': [
                {
                    'text': self._preprocess_text(utterance['text']),
                    'speaker': WizardOfWikipedia._convert_speaker_id(utterance['speaker'])
                }
                for utterance in dialogue['dialog']
            ]
        }

        return dialogue

    def _load_samples(self) -> List[Dict[str, Union[str, Dict[str, str]]]]:
        # Get split file name
        if self.split == 'train':
            file_name = 'train.json'
        elif self.split == 'validation':
            file_name = 'valid_random_split.json'
        elif self.split == 'test':
            file_name = 'test_random_split.json'
        else:
            raise ValueError(f'Unknown value for data set split: {self.split}')
        # Load split of the corpus
        with open(os.path.join(self.corpus_dir_path, file_name)) as f:
            dialogues = json.loads(f.read())
        # Apply subsampling if required
        if self.sample is not None:
            # Get number of samples to collect
            n_samples = self.sample if isinstance(self.sample, int) else int(
                math.ceil(self.sample * len(dialogues)))
            # Subsample data set unless the number of samples to take is equal to the number of samples available
            if n_samples != len(dialogues):
                dialogues = random.sample(dialogues, n_samples)
        # Standardise corpus
        with parallel_backend(self.joblib_backend, n_jobs=self.n_jobs):
            return Parallel(verbose=self.verbosity_level)(
                delayed(self._preprocess_dialogue)(dialogue) for dialogue in dialogues
            )

    @classmethod
    def _get_labels(
            cls,
            dialogue: Dict,
            augmentation: bool,
            global_labels_metadata: Optional[Dict[str, str]],
            line_labels_metadata: Optional[Dict[str, str]]
    ) -> Tuple[Optional[Dict[str, str]], Optional[List[Dict[str, str]]]]:
        if global_labels_metadata is not None and len(global_labels_metadata) > 0:
            global_labels = {'topic': dialogue['topic']}
        else:
            global_labels = None
        line_labels = None

        return global_labels, line_labels


class IEMOCAP(_DialogueCorpus):
    # Corpus metadata
    IDENTIFIER: str = 'IEMOCAP_full_release'
    LINE_LABELS_METADATA: Dict[str, Dict[str, Optional[str]]] = {
        'emotion': {
            'id': 'emotion',
            'description': 'The emotion is the categorical affective state characterising '
                           'a speakers during the interaction. '
                           'The emotion can be recognised by what the speaker is saying.',
            'values': [
                'neutral state',
                'happiness',
                'sadness',
                'anger',
                'surprise',
                'fear',
                'disgust',
                'frustration',
                'excited',
                'other'
            ]
        },
        'valence': {
            'id': 'valence',
            'description': 'The valence (or appraisal) describes the extent to which '
                           'an emotion is pleasant or unpleasant. '
                           'Valence is one of the possible dimensions of the emotion space and can be '
                           'recognised by what a speaker is saying.',
            'values': ['very negative', 'negative', 'neutral', 'positive', 'very positive']
        },
        'activation': {
            'id': 'activation',
            'description': 'The activation (or arousal) describes the intensity,(in the sense of strength)'
                           ' of the associated emotional state . '
                           'Activation is one of the possible dimensions of the emotion space and can be '
                           'recognised by what a speaker is saying.',
            'values': ['very low', 'low', 'medium', 'high', 'very high']
        },
        'dominance': {
            'id': 'dominance',
            'description': 'The dominance (or power) is the apparent strength of a speaker in a conversation and '
                           'describes the ability to handle a situation. '
                           'Dominance is one of the possible dimensions of the emotion space and can be '
                           'recognised by what a speaker is saying.',
            'values': ['very weak', 'weak', 'medium', 'strong', 'very strong']
        }
    }
    # Data loading and preprocessing parameters
    LINE_REGEX: Pattern[str] = re.compile(
        r'^((Ses(\d+)[MF]_(impro|script)\d+(_?\d\d*\w*)?)_([MF])\d+) \[\d{3}\.\d{4}-\d{3}\.\d{4}\]: (.+)$'
    )
    ANNOTATION_REGEX: Pattern[str] = re.compile(
        r'^\[\d+\.\d+ - \d+\.\d+]\t(\w+)\t(\w+)\t\[(\d\.\d+), (\d\.\d+), (\d\.\d+)]$'
    )
    CATEGORICAL_EMOTION_DECODER: Dict[str, str] = {
        'neu': 'neutral state',
        'hap': 'happiness',
        'sad': 'sadness',
        'ang': 'anger',
        'sur': 'surprise',
        'fea': 'fear',
        'dis': 'disgust',
        'fru': 'frustration',
        'exc': 'excited',
        'oth': 'other'
    }
    SFX: List[str] = ['[LAUGHTER]', '[LIPSMACK]', '[GARBAGE]', '[BREATHING]']

    def _preprocess_text(self, text: str) -> str:
        for sfx in self.SFX:
            text = text.replace(sfx, '')
        return super(IEMOCAP, self)._preprocess_text(text)

    def _parse_line(self, raw_data) -> Dict[str, str]:
        # Split expression with regex
        identifier, *_, text = self.LINE_REGEX.findall(raw_data)[0]
        # Build dictionary
        parsed_data = {'text': text, 'id': identifier}

        return parsed_data

    def _parse_dialogue(self, raw_data: str) -> List[Dict[str, str]]:
        # Break dialogue into lines
        lines = raw_data.split('\n')
        # Build list of utterances
        dialogue: List[Dict[str, str]] = [self._parse_line(line) for line in lines if self.LINE_REGEX.match(line)]

        return dialogue

    def _parse_annotation(self, raw_data) -> Dict[str, str]:
        # Split expression with regex
        *_, emotion, valence, activation, dominance = self.ANNOTATION_REGEX.findall(raw_data)[0]
        # Build dictionary
        parsed_data = {
            'emotion': self.CATEGORICAL_EMOTION_DECODER.get(emotion, 'neutral state'),
            'valence': self.LINE_LABELS_METADATA['valence']['values'][min(4, max(0, int(math.ceil(float(valence))) - 1))],
            'activation': self.LINE_LABELS_METADATA['activation']['values'][min(4, max(0, int(math.ceil(float(activation))) - 1))],
            'dominance': self.LINE_LABELS_METADATA['dominance']['values'][min(4, max(0, int(math.ceil(float(dominance))) - 1))]
        }

        return parsed_data

    def _parse_annotations(self, raw_data) -> Dict[str, Dict[str, str]]:
        # Retrieve lines from file
        raw_annotations: List[str] = [annotation.split('\n')[0] for annotation in raw_data.split('\n\n')[1:]]
        # Parse all the lines
        annotations: Dict[str, Dict[str, str]] = {
            self.ANNOTATION_REGEX.search(raw_annotation).group(1) : self._parse_annotation(raw_annotation)
            for raw_annotation in raw_annotations if self.ANNOTATION_REGEX.match(raw_annotation)
        }

        return annotations

    def _preprocess_dialogue(
            self, text_file_path: str, annotation_file_path: str
    ) -> Dict[str, Union[str, Dict[str, str]]]:
        # Load dialogue data
        with open(text_file_path) as f:
            raw_dialogue = f.read().strip()
        # Parse lines
        dialogue: List[Dict[str, str]] = self._parse_dialogue(raw_dialogue)
        # Load dialogue annotations
        with open(annotation_file_path) as f:
            raw_annotations = f.read().strip()
        annotations: Dict[str, Dict[str, str]] = self._parse_annotations(raw_annotations)

        assert len(dialogue) == len(annotations), f'\'{text_file_path}\' \'{annotation_file_path}\''

        # Pre-process dialogue
        dialogue: Dict[str, Union[str, Dict[str, str]]] = {
            'split': self.split,
            'corpus': 'IEMOCAP',
            'utterances': [
                {'text': self._preprocess_text(utterance['text']), **annotations[utterance['id']]}
                for utterance in dialogue
            ]
        }

        return dialogue

    def _load_samples(self) -> List[Dict[str, Union[str, Dict[str, str]]]]:
        # NOTE the data set does not provide train-validation-test split
        # Get file list
        dialogue_file_paths: List[Tuple[str, str]] = [
            (
                os.path.join(self.corpus_dir_path, ses_dir, 'dialog', 'transcriptions', transcripts_file),
                os.path.join(self.corpus_dir_path, ses_dir, 'dialog', 'EmoEvaluation', transcripts_file)
            )
            for ses_dir in (
                ses_dir for ses_dir in os.listdir(self.corpus_dir_path)
                if ses_dir.startswith('Session') and os.path.isdir(os.path.join(self.corpus_dir_path, ses_dir))
            )
            for transcripts_file in os.listdir(os.path.join(self.corpus_dir_path, ses_dir, 'dialog', 'transcriptions'))
            if transcripts_file.endswith('.txt')
        ]
        # Get indices list
        idxs = range(len(dialogue_file_paths))
        # Do train-validation-test split on the indices
        train_idxs, test_idxs = train_test_split(idxs, test_size=self.holdout, random_state=self.random_seed)
        train_idxs, validation_idxs = train_test_split(train_idxs, test_size=self.holdout, random_state=self.random_seed)
        # Get list of current split indices
        if self.split == 'train':
            idxs = train_idxs
        elif self.split == 'validation':
            idxs = validation_idxs
        elif self.split == 'test':
            idxs = test_idxs
        else:
            raise ValueError(f'Unknown value for data set split: {self.split}')
        # Select file paths from current split
        dialogue_file_paths = [dialogue_file_paths[i] for i in idxs]
        # Apply subsampling if required
        if self.sample is not None:
            # Get number of samples to collect
            if isinstance(self.sample, int):
                n_samples = self.sample
            else:
                n_samples = int(math.ceil(self.sample * len(dialogue_file_paths)))
            # Subsample data set unless the number of samples to take is equal to the number of samples available
            if n_samples != len(dialogue_file_paths):
                dialogue_file_paths = random.sample(dialogue_file_paths, n_samples)
        # Standardise corpus
        with parallel_backend(self.joblib_backend, n_jobs=self.n_jobs):
            return Parallel(verbose=self.verbosity_level)(
                delayed(self._preprocess_dialogue)(text_file_path, annotation_file_path)
                for text_file_path, annotation_file_path in dialogue_file_paths
            )

    @classmethod
    def _get_labels(
            cls,
            dialogue: Dict,
            augmentation: bool,
            global_labels_metadata: Optional[Dict[str, str]],
            line_labels_metadata: Optional[Dict[str, str]]
    ) -> Tuple[Optional[Dict[str, str]], Optional[List[Dict[str, str]]]]:
        global_labels = None
        if line_labels_metadata is not None and len(line_labels_metadata) > 0:
            line_labels = [
                {
                    'emotion': utterance['emotion'],
                    'valence': utterance['valence'],
                    'activation': utterance['activation'],
                    'dominance': utterance['dominance']
                }
                for utterance in dialogue['utterances']
            ]
        else:
            line_labels = None

        return global_labels, line_labels


class TopicalChat(_DialogueCorpus):
    # Corpus metadata
    IDENTIFIER = 'Topical-Chat-master'

    def _preprocess_dialogue(self, dialogue: Dict) -> Dict[str, Union[str, Dict[str, str]]]:
        # TODO add topic and sentiment
        # Pre-processed dialogue
        dialogue: Dict[str, Union[str, Dict[str, str]]] = {
            'split': self.split,
            'corpus': 'Topical-Chat',
            'utterances': [
                {
                    'text': self._preprocess_text(utterance['message'])
                } for utterance in dialogue['content']
            ]
        }

        return dialogue

    def _load_samples(self) -> List[Dict[str, Union[str, Dict[str, str]]]]:
        # Get split file name
        if self.split == 'train':
            file_names = ['train.json']
        elif self.split == 'validation':
            file_names = ['valid_freq.json', 'valid_rare.json']
        elif self.split == 'test':
            file_names = ['test_freq.json', 'test_rare.json']
        else:
            raise ValueError(f'Unknown value for data set split: {self.split}')
        # Load split of the corpus
        dialogues = dict()
        for file_name in file_names:
            with open(os.path.join(self.corpus_dir_path, 'conversations', file_name)) as f:
                dialogues |= json.load(f)
        # Apply subsampling if required
        if self.sample is not None:
            # Get number of samples to collect
            n_samples = self.sample if isinstance(self.sample, int) else int(math.ceil(self.sample * len(dialogues)))
            # Subsample data set unless the number of samples to take is equal to the number of samples available
            if n_samples != len(dialogues):
                dialogues = {
                    dialogue_key: dialogues[dialogue_key] for dialogue_key in random.sample(list(dialogues), n_samples)
                }
        # Standardise corpus
        with parallel_backend(self.joblib_backend, n_jobs=self.n_jobs):
            return Parallel(verbose=self.verbosity_level)(
                delayed(self._preprocess_dialogue)(dialogues[dialogue_key]) for dialogue_key in dialogues
            )


# Therapy data sets

class CounsellingAndPsychotherapyCorpus(_DialogueCorpus):
    # Corpus metadata
    IDENTIFIER = 'Counseling_and_Psychotherapy_Transcripts_Volume_II'
    # Composable strings to generate the dialogue
    INTERACTION_TYPE: List[str] = ['conversation', 'dialogue', 'chit-chat']
    CHUNK_TYPE: List[str] = ['window', 'chunk', 'piece', 'passage']
    CONTEXT_TYPE: List[str] = ['context', 'history', 'past']
    RESPONSE_TYPE: List[str] = ['response', 'utterance']
    # Premise
    PREMISE: List[str] = [
        f'The following is a therapy session between {SPEAKER_DESCRIPTIONS}.',
        f'The following is a therapy {INTERACTION} between {SPEAKER_DESCRIPTIONS}.',
        f'Here follows a therapy {INTERACTION} between {SPEAKER_DESCRIPTIONS}.',
        f'This is a therapy {INTERACTION} between {SPEAKER_DESCRIPTIONS}.',
        f'The following is the transcript of a therapy {INTERACTION} between {SPEAKER_DESCRIPTIONS}.',
        f'The following is a counselling session between {SPEAKER_DESCRIPTIONS}.',
        f'The following is a counselling {INTERACTION} between {SPEAKER_DESCRIPTIONS}.',
        f'Here follows a counselling {INTERACTION} between {SPEAKER_DESCRIPTIONS}.',
        f'This is a counselling {INTERACTION} between {SPEAKER_DESCRIPTIONS}.',
        f'The following is the transcript of a counselling {INTERACTION} between {SPEAKER_DESCRIPTIONS}.'
    ]
    THERAPIST_HUMAN_PARTICIPANT: List[str] = ['a therapist', 'a counsellor', 'a mental health professional']
    PATIENT_HUMAN_PARTICIPANT: List[str] = ['a person', 'a client']
    THERAPY_BOT_PARTICIPANT: List[str] = [
        'an AI for therapy', 'a chatbot for therapy', 'a conversational agent for therapy', 'an dialogue agent for therapy'
        'an AI simulating a therapist', 'an AI', 'an AI system simulating a therapist',
        'an AI system', 'a chatbot simulating a therapist', 'a chatbot',
        'a conversational agent simulating a therapist', 'a conversational agent',
        'a dialogue agent simulating a therapist', 'a dialogue agent'
    ]
    PATIENT_BOT_PARTICIPANT: List[str] = [
        'an AI simulating a therapy patient', 'an AI', 'an AI system simulating a therapy patient',
        'an AI system', 'a chatbot simulating a therapy patient', 'a chatbot',
        'a conversational agent simulating a therapy patient', 'a conversational agent',
        'an dialogue agent simulating a therapy patient', 'an dialogue agent'
    ]
    DISCRIMINATIVE_TASK_PROMPT: Dict[str, List[str]] = {
        'global': [
            f'{LABEL_ID}{CLS_SEP}',
            f'{LABEL_ID} in the {INTERACTION} {DIALOGUE_CHUNK} {CLS_SEP}',
            f'The {LABEL_ID} in the {INTERACTION} {DIALOGUE_CHUNK} is'
        ],
        'local': [f'{LABEL_ID}{CLS_SEP}', f'{RESPONSE} {LABEL_ID}{CLS_SEP}', f'The {LABEL_ID} of the {RESPONSE} is']
    }
    GENERATIVE_TASK_PREFIX: List[str] = [
        f'In the following interactions, {SPEAKERS} will converse in natural language. '
        f'The {PATIENT} talks about his/hers issues to the {THERAPIST} '
        f'and the {THERAPIST} helps the {PATIENT} explore and solve his/hers problems.',
        f'During these interactions, {SPEAKERS} will converse in natural language. '
        f'The {PATIENT} talks about his/hers issues to the {THERAPIST} '
        f'and the {THERAPIST} helps the {PATIENT} explore and solve his/hers problems.',
        f'In the following interactions, {SPEAKERS} converse in natural language. '
        f'The {PATIENT} talks about his/hers issues to the {THERAPIST} '
        f'and the {THERAPIST} helps the {PATIENT} explore and solve his/hers problems.',
        f'During these interactions, {SPEAKERS} converse in natural language. '
        f'The {PATIENT} talks about his/hers issues to the {THERAPIST} '
        f'and the {THERAPIST} helps the {PATIENT} explore and solve his/hers problems.',
        f'In the following interactions, {SPEAKERS} are conversing in natural language. '
        f'The {PATIENT} talks about his/hers issues to the {THERAPIST} '
        f'and the {THERAPIST} helps the {PATIENT} explore and solve his/hers problems.',
        f'During these interactions, {SPEAKERS} are conversing in natural language. '
        f'The {PATIENT} talks about his/hers issues to the {THERAPIST} '
        f'and the {THERAPIST} helps the {PATIENT} explore and solve his/hers problems.'
    ]
    # IDs
    HUMAN_SPEAKER_IDS: Dict[str, List[Tuple[str, str]]] = {
        'default': [
            ('Therapist', 'Patient'),
            ('Counsellor', 'Patient'),
            ('Therapist', 'Client'),
            ('Counsellor', 'Client')
        ]
    }
    HUMAN_THERAPY_BOT_SPEAKER_IDS: Dict[str, List[Tuple[str, str]]] = {
        'AI': [
            ('AI', 'Patient'),
            ('AI', 'Client'),
            ('AI', 'User'),
            ('Therapy AI', 'Patient'),
            ('Therapy AI', 'Client'),
            ('Therapy AI', 'User'),
            ('Counselling AI', 'Patient'),
            ('Counselling AI', 'Client'),
            ('Counselling AI', 'User')
        ],
        'chatbot': [
            ('Chatbot', 'Patient'),
            ('Chatbot', 'Client'),
            ('Chatbot', 'User'),
            ('Therapy Chatbot', 'Patient'),
            ('Therapy Chatbot', 'Client'),
            ('Therapy Chatbot', 'User'),
            ('Counselling Chatbot', 'Patient'),
            ('Counselling Chatbot', 'Client'),
            ('Counselling Chatbot', 'User')
        ]
    }
    HUMAN_PATIENT_BOT_SPEAKER_IDS: Dict[str, List[Tuple[str, str]]] = {
        'AI': [
            ('Therapist', 'AI'),
            ('Counsellor', 'AI'),
            ('Therapist', 'Client AI'),
            ('Counsellor', 'Client AI'),
            ('Therapist', 'Patient AI'),
            ('Counsellor', 'Patient AI')
        ],
        'chatbot': [
            ('Therapist', 'Chatbot'),
            ('Counsellor', 'Chatbot'),
            ('Therapist', 'Client Chatbot'),
            ('Counsellor', 'Client Chatbot'),
            ('Therapist', 'Patient Chatbot'),
            ('Counsellor', 'Patient Chatbot')
        ],
    }
    # Data loading and preprocessing parameters
    LINE_REGEX: Pattern[str] = re.compile(r'^(CLIENT|THERAPIST): .+$')
    ROLES_DECODER: Dict = {'THERAPIST': 'therapist', 'CLIENT': 'patient'}

    def _preprocess_text(self, text: str) -> str:
        text = text.replace('<|unknown|>', '###')
        return super(CounsellingAndPsychotherapyCorpus, self)._preprocess_text(text)

    def _parse_line(self, raw_data) -> Dict[str, str]:
        # Split expression with regex
        speaker, text = raw_data.split(': ', 1)
        # Build dictionary
        parsed_data = {'speaker': self.ROLES_DECODER[speaker], 'text': text}

        return parsed_data

    def _parse_dialogue(self, raw_data: str) -> List[Dict[str, str]]:
        # Break dialogue into lines
        lines = raw_data.split('\n')
        # Build list of utterances
        dialogue: List[Dict[str, str]] = [self._parse_line(line) for line in lines]

        return dialogue

    def _preprocess_dialogue(self, dialogue_file_path: str) -> Optional[Dict[str, Union[str, Dict[str, str]]]]:
        # Load dialogue data
        with open(dialogue_file_path) as f:
            raw_dialogue = f.read().strip()
        # Integrity check
        if not all(self.LINE_REGEX.match(line) for line in raw_dialogue.split('\n')):
            return None
        # Parse lines
        dialogue: List[Dict[str, str]] = self._parse_dialogue(raw_dialogue)
        # Pre-processed dialogue
        dialogue: Dict[str, Union[str, Dict[str, str]]] = {
            'split': self.split,
            'corpus': 'Counselling and Psychotherapy Transcripts Volume II',
            'utterances': [
                {'text': self._preprocess_text(utterance['text']), 'speaker': utterance['speaker']}
                for utterance in dialogue
            ]
        }

        return dialogue

    def _load_samples(self) -> List[Dict[str, Union[str, Dict[str, str]]]]:
        # NOTE the data set does not provide train-validation-test split
        # Get file list
        dialogue_file_paths: List[str] = [
            os.path.join(self.corpus_dir_path, transcripts_file)
            for transcripts_file in os.listdir(self.corpus_dir_path) if transcripts_file.endswith('.txt')
        ]
        # Get indices list
        idxs = range(len(dialogue_file_paths))
        # Do train-validation-test split on the indices
        train_idxs, test_idxs = train_test_split(idxs, test_size=self.holdout, random_state=self.random_seed)
        train_idxs, validation_idxs = train_test_split(train_idxs, test_size=self.holdout, random_state=self.random_seed)
        # Get list of current split indices
        if self.split == 'train':
            idxs = train_idxs
        elif self.split == 'validation':
            idxs = validation_idxs
        elif self.split == 'test':
            idxs = test_idxs
        else:
            raise ValueError(f'Unknown value for data set split: {self.split}')
        # Select file paths from current split
        dialogue_file_paths = [dialogue_file_paths[i] for i in idxs]
        # Apply subsampling if required
        if self.sample is not None:
            # Get number of samples to collect
            if isinstance(self.sample, int):
                n_samples = self.sample
            else:
                n_samples = int(math.ceil(self.sample * len(dialogue_file_paths)))
            # Subsample data set unless the number of samples to take is equal to the number of samples available
            if n_samples != len(dialogue_file_paths):
                dialogue_file_paths = random.sample(dialogue_file_paths, n_samples)
        # Standardise corpus
        with parallel_backend(self.joblib_backend, n_jobs=self.n_jobs):
             dialogues = Parallel(verbose=self.verbosity_level)(
                delayed(self._preprocess_dialogue)(dialogue_file_path) for dialogue_file_path in dialogue_file_paths
            )

        return [dialogue for dialogue in dialogues if dialogue is not None]

    @classmethod
    def _get_speakers(
            cls, dialogue: Dict, augmentation: bool
    ) -> Tuple[Tuple[str, str], bool]:
        therapist_fist = dialogue['utterances'][0]['speaker'] == 'therapist'
        if augmentation:
            bot = random.choice([True, False])
            if bot:
                if random.choice([True, False]):
                    sp1, sp2 = random.choice(cls.HUMAN_THERAPY_BOT_SPEAKER_IDS[random.choice(list(cls.HUMAN_BOT_SPEAKER_IDS))])
                else:
                    sp1, sp2 = random.choice(cls.HUMAN_PATIENT_BOT_SPEAKER_IDS[random.choice(list(cls.HUMAN_BOT_SPEAKER_IDS))])
            else:
                sp1, sp2 = random.choice(cls.HUMAN_SPEAKER_IDS[random.choice(list(cls.HUMAN_SPEAKER_IDS))])
        else:
            sp1, sp2 = cls.HUMAN_SPEAKER_IDS['default'][0]
            bot = False

        speakers = (sp1, sp2) if therapist_fist else (sp2, sp1)

        return speakers, bot

    @classmethod
    def _compose_premise(cls, speakers: Tuple[str, str], interaction: str, augmentation: bool, bot: bool) -> str:
        # Presentation line
        presentation = random.choice(cls.PRESENTATION) if augmentation else cls.PRESENTATION[0]
        # Generate speakers introduction and description
        if speakers[0].lower() in {'patient', 'user', 'client'} or any(role in speakers[0].lower() for role in {'patient', 'user', 'client'}):
            spp, spt = speakers
            speaker_first = False
        elif speakers[1].lower() in {'patient', 'user', 'client'} or any(role in speakers[1].lower() for role in {'patient', 'user', 'client'}):
            spt, spp = speakers
            speaker_first = True
        elif speakers[0].lower() in {'therapist', 'counsellor'} or any(role in speakers[0].lower() for role in {'therapist', 'counsellor'}):
            spt, spp = speakers
            speaker_first = True
        else:
            spp, spt = speakers
            speaker_first = False
        if bot:
            if any(spt.lower() in key.lower() or key.lower() in spt.lower() for key in cls.HUMAN_THERAPY_BOT_SPEAKER_IDS):
                pt_list, pp_list = cls.THERAPY_BOT_PARTICIPANT, cls.PATIENT_HUMAN_PARTICIPANT
            else:
                pt_list, pp_list = cls.THERAPIST_HUMAN_PARTICIPANT, cls.PATIENT_BOT_PARTICIPANT
        else:
            pt_list, pp_list = cls.THERAPIST_HUMAN_PARTICIPANT, cls.PATIENT_HUMAN_PARTICIPANT
        pt = random.choice(pt_list) if augmentation else pt_list[0]
        pp = random.choice(pp_list) if augmentation else pp_list[0]
        if speaker_first:
            speakers_description = f'{pt}, {presentation} {spt}, and {pp}, {presentation} {spp}'
        else:
            speakers_description = f'{pp}, {presentation} {spp}, and {pt}, {presentation} {spt}'

        # Compose premise
        premise = random.choice(cls.PREMISE) if augmentation else cls.PREMISE[0]
        premise = premise.replace(INTERACTION, interaction)
        premise = premise.replace(SPEAKER_DESCRIPTIONS, speakers_description)
        premise = premise.replace(THERAPIST, spt)
        premise = premise.replace(PATIENT, spp)

        return premise

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
        if speakers[0].lower() in {'patient', 'user', 'client'} or any(role in speakers[0].lower() for role in {'patient', 'user', 'client'}):
            spp, spt = speakers
        elif speakers[1].lower() in {'patient', 'user', 'client'} or any(role in speakers[1].lower() for role in {'patient', 'user', 'client'}):
            spt, spp = speakers
        elif speakers[0].lower() in {'therapist', 'counsellor'} or any(role in speakers[0].lower() for role in {'therapist', 'counsellor'}):
            spt, spp = speakers
        else:
            spp, spt = speakers
        task_description = super(CounsellingAndPsychotherapyCorpus, cls)._compose_task_description(
            speakers, model_type, interaction, augmentation, dropout, bot,
            global_labels_metadata=global_labels_metadata, line_labels_metadata=line_labels_metadata,
            tgt_labels_metadata=tgt_labels_metadata, label_type=label_type, chunk=chunk, context=context,
            response=response
        )
        task_description = task_description.replace(THERAPIST, spt)
        task_description = task_description.replace(PATIENT, spp)

        return task_description


class HOPE(_DialogueCorpus):
    # Corpus metadata
    IDENTIFIER = 'HOPE_WSDM_2022'
    LINE_LABELS_METADATA: Dict[str, Dict[str, Optional[str]]] = {
        'dialogue_act_category': {
            'id': 'dialogue act category',
            'description': 'Dialogue act categories describe at high level the '
                           'requirements of an utterance in a dialogue.'
                           'They are used to capture the intents of the speaker at a coarse grained level',
            'values': ['speaker initiative', 'general', 'speaker responsive']
        },
        'dialogue_act': {
            'id': 'dialogue act',
            'description': 'Dialogue acts allow to understand the intended '
                           'requirements of an utterance in a conversation.'
                           'They are used to capture the intents of the speaker at a fine grained level',
            'values': [
                'clarification request',
                'opinion request',
                'information request',
                'yes/no question',
                'greeting',
                'acknowledgement',
                'general chat',
                'information delivery',
                'clarification delivery',
                'opinion delivery',
                'positive answer',
                'negative answer'
            ]
        },
    }
    # Composable strings to generate the dialogue
    INTERACTION_TYPE: List[str] = ['conversation', 'dialouge', 'chit-chat']
    CHUNK_TYPE: List[str] = ['window', 'chunk', 'piece', 'passage']
    CONTEXT_TYPE: List[str] = ['context', 'history', 'past']
    RESPONSE_TYPE: List[str] = ['response', 'utterance']
    # Premise
    PREMISE: List[str] = [
        f'The following is a therapy session between {SPEAKER_DESCRIPTIONS}.',
        f'The following is a therapy {INTERACTION} between {SPEAKER_DESCRIPTIONS}.',
        f'Here follows a therapy {INTERACTION} between {SPEAKER_DESCRIPTIONS}.',
        f'This is a therapy {INTERACTION} between {SPEAKER_DESCRIPTIONS}.',
        f'The following is the transcript of a therapy {INTERACTION} between {SPEAKER_DESCRIPTIONS}.',
        f'The following is a counselling session between {SPEAKER_DESCRIPTIONS}.',
        f'The following is a counselling {INTERACTION} between {SPEAKER_DESCRIPTIONS}.',
        f'Here follows a counselling {INTERACTION} between {SPEAKER_DESCRIPTIONS}.',
        f'This is a counselling {INTERACTION} between {SPEAKER_DESCRIPTIONS}.',
        f'The following is the transcript of a counselling {INTERACTION} between {SPEAKER_DESCRIPTIONS}.'
    ]
    THERAPIST_HUMAN_PARTICIPANT: List[str] = ['a therapist', 'a counsellor', 'a mental health professional']
    PATIENT_HUMAN_PARTICIPANT: List[str] = ['a person', 'a client']
    THERAPY_BOT_PARTICIPANT: List[str] = [
        'an AI for therapy', 'a chatbot for therapy', 'a conversational agent for therapy',
        'an dialogue agent for therapy'
        'an AI simulating a therapist', 'an AI', 'an AI system simulating a therapist',
        'an AI system', 'a chatbot simulating a therapist', 'a chatbot',
        'a conversational agent simulating a therapist', 'a conversational agent',
        'an dialogue agent simulating a therapist', 'an dialogue agent'
    ]
    PATIENT_BOT_PARTICIPANT: List[str] = [
        'an AI simulating a therapy patient', 'an AI', 'an AI system simulating a therapy patient',
        'an AI system', 'a chatbot simulating a therapy patient', 'a chatbot',
        'a conversational agent simulating a therapy patient', 'a conversational agent',
        'a dialogue agent simulating a therapy patient', 'a dialogue agent'
    ]
    DISCRIMINATIVE_TASK_PROMPT: Dict[str, List[str]] = {
        'global': [
            f'{LABEL_ID}{CLS_SEP}',
            f'{LABEL_ID} in the {INTERACTION} {DIALOGUE_CHUNK} {CLS_SEP}',
            f'The {LABEL_ID} in the {INTERACTION} {DIALOGUE_CHUNK} is'
        ],
        'local': [f'{LABEL_ID}{CLS_SEP}', f'{RESPONSE} {LABEL_ID}{CLS_SEP}', f'The {LABEL_ID} of the {RESPONSE} is']
    }
    GENERATIVE_TASK_PREFIX: List[str] = [
        f'In the following interactions, {SPEAKERS} will converse in natural language. '
        f'The {PATIENT} talks about his/hers issues to the {THERAPIST} '
        f'and the {THERAPIST} helps the {PATIENT} explore and solve his/hers problems.',
        f'During these interactions, {SPEAKERS} will converse in natural language. '
        f'The {PATIENT} talks about his/hers issues to the {THERAPIST} '
        f'and the {THERAPIST} helps the {PATIENT} explore and solve his/hers problems.',
        f'In the following interactions, {SPEAKERS} converse in natural language. '
        f'The {PATIENT} talks about his/hers issues to the {THERAPIST} '
        f'and the {THERAPIST} helps the {PATIENT} explore and solve his/hers problems.',
        f'During these interactions, {SPEAKERS} converse in natural language. '
        f'The {PATIENT} talks about his/hers issues to the {THERAPIST} '
        f'and the {THERAPIST} helps the {PATIENT} explore and solve his/hers problems.',
        f'In the following interactions, {SPEAKERS} are conversing in natural language. '
        f'The {PATIENT} talks about his/hers issues to the {THERAPIST} '
        f'and the {THERAPIST} helps the {PATIENT} explore and solve his/hers problems.',
        f'During these interactions, {SPEAKERS} are conversing in natural language. '
        f'The {PATIENT} talks about his/hers issues to the {THERAPIST} '
        f'and the {THERAPIST} helps the {PATIENT} explore and solve his/hers problems.'
    ]
    # IDs
    HUMAN_SPEAKER_IDS: Dict[str, List[Tuple[str, str]]] = {
        'default': [
            ('Therapist', 'Patient'),
            ('Counsellor', 'Patient'),
            ('Therapist', 'Client'),
            ('Counsellor', 'Client')
        ]
    }
    HUMAN_THERAPY_BOT_SPEAKER_IDS: Dict[str, List[Tuple[str, str]]] = {
        'AI': [
            ('AI', 'Patient'),
            ('AI', 'Client'),
            ('AI', 'User'),
            ('Therapy AI', 'Patient'),
            ('Therapy AI', 'Client'),
            ('Therapy AI', 'User'),
            ('Counselling AI', 'Patient'),
            ('Counselling AI', 'Client'),
            ('Counselling AI', 'User')
        ],
        'chatbot': [
            ('Chatbot', 'Patient'),
            ('Chatbot', 'Client'),
            ('Chatbot', 'User'),
            ('Therapy Chatbot', 'Patient'),
            ('Therapy Chatbot', 'Client'),
            ('Therapy Chatbot', 'User'),
            ('Counselling Chatbot', 'Patient'),
            ('Counselling Chatbot', 'Client'),
            ('Counselling Chatbot', 'User')
        ]
    }
    HUMAN_PATIENT_BOT_SPEAKER_IDS: Dict[str, List[Tuple[str, str]]] = {
        'AI': [
            ('Therapist', 'AI'),
            ('Counsellor', 'AI'),
            ('Therapist', 'Client AI'),
            ('Counsellor', 'Client AI'),
            ('Therapist', 'Patient AI'),
            ('Counsellor', 'Patient AI')
        ],
        'chatbot': [
            ('Therapist', 'Chatbot'),
            ('Counsellor', 'Chatbot'),
            ('Therapist', 'Client Chatbot'),
            ('Counsellor', 'Client Chatbot'),
            ('Therapist', 'Patient Chatbot'),
            ('Counsellor', 'Patient Chatbot')
        ],
    }
    # Data loading and preprocessing parameters
    FILE_NAME_REGEX: Pattern[str] = re.compile(r'^Copy of \d+\.csv$')
    FINE_DIALOGUE_ACT_DECODER: Dict = {
        'acak': 'acknowledgement',
        'ack': 'acknowledgement',
        # 'ap': '',  # 2 # Answer positive?
        # 'ay': '',  # 2 # Answer yes?
        'cd': 'clarification delivery',
        'cdd': 'clarification delivery',
        'ci': 'general chat',  # 50 # Maybe??
        # 'com': '',  # 9
        # 'comp': '',  # 2
        'cq': 'clarification delivery',  # Typo?
        'cr': 'general chat',  # 110 # Maybe??
        'crq': 'clarification request',
        'cv': 'general chat',  # 61 # Maybe??
        'gc': 'general chat',
        'gt': 'greeting',
        # 'hp': '',  # 1
        'id': 'information delivery',
        'in': 'information delivery',  # Typo?
        'irq': 'information request',
        'irrq': 'information request',
        'o': 'opinion delivery',  # Opinion?
        'od': 'opinion delivery',
        'on': 'negative answer',  # Opinion Negative?
        'op': 'positive answer',  # Opinion Positive?
        'orq': 'opinion request',
        'urq': 'opinion request',  # Typo?
        # 'vc': '',  # 1
        # 'yk': '',  # 1
        'yq': 'yes/no question'
    }
    COARSE_DIALOGUE_ACT_DECODER: Dict = {
        'acak': 'general',
        'ack': 'general',
        # 'ap': '',  # 2 # Answer positive?
        # 'ay': '',  # 2 # Answer yes?
        'cd': 'speaker responsive',
        'cdd': 'speaker responsive',
        'ci': 'general',  # 50 # Maybe??
        # 'com': '',  # 9
        # 'comp': '',  # 2
        'cq': 'speaker responsive',  # Typo?
        'cr': 'general',  # 110 # Maybe??
        'crq': 'speaker initiative',
        'cv': 'general',  # 61 # Maybe??
        'gc': 'general',
        'gt': 'general',
        # 'hp': '',  # 1
        'id': 'speaker responsive',
        'in': 'speaker responsive',  # Typo?
        'irq': 'speaker initiative',
        'irrq': 'speaker initiative',
        'o': 'speaker responsive',  # Opinion?
        'od': 'speaker responsive',
        'on': 'speaker responsive',  # Opinion Negative?
        'op': 'speaker responsive',  # Opinion Positive?
        'orq': 'speaker initiative',
        'urq': 'speaker initiative',  # Typo?
        # 'vc': '',  # 1
        # 'yk': '',  # 1
        'yq': 'speaker initiative'
    }
    ACTIONS_DECODER: Dict = {
        'acak': 'acknowledgement',
        'ack': 'acknowledgement',
        # 'ap': '',  # 2 # Answer positive?
        # 'ay': '',  # 2 # Answer yes?
        'cd': 'clarification delivery',
        'cdd': 'clarification delivery',
        'ci': 'general chat',  # 50 # Maybe??
        # 'com': '',  # 9
        # 'comp': '',  # 2
        'cq': 'clarification delivery',  # Typo?
        'cr': 'general chat',  # 110 # Maybe??
        'crq': 'clarification request',
        'cv': 'general chat',  # 61 # Maybe??
        'gc': 'general chat',
        'gt': 'greeting',
        # 'hp': '',  # 1
        'id': 'information delivery',
        'in': 'information delivery',  # Typo?
        'irq': 'information request',
        'irrq': 'information request',
        'o': 'opinion delivery',  # Opinion?
        'od': 'opinion delivery',
        'on': 'negative answer',  # Opinion Negative?
        'op': 'positive answer',  # Opinion Positive?
        'orq': 'opinion request',
        'urq': 'opinion request',  # Typo?
        # 'vc': '',  # 1
        # 'yk': '',  # 1
        'yq': 'yes/no question'
    }
    ACTION_REGEX: Pattern[str] = re.compile(r'(\w+)([^\w\n]+\w*)*')
    SPEAKER_DECODER: Dict = {'T': 'therapist', 'P': 'patient'}

    def _preprocess_dialogue(self, dialogue_file_path: str) -> Dict[str, Union[str, Dict[str, str]]]:
        # Load dialogue data
        df = pd.read_csv(dialogue_file_path)
        # Get identifier for dialogue act
        if 'Dialogue Act' in df.columns:
            dialogue_act_col = 'Dialogue Act'
        elif 'Dialogue_Act' in df.columns:
            dialogue_act_col = 'Dialogue_Act'
        elif 'Dialog_Act' in df.columns:
            dialogue_act_col = 'Dialog_Act'
        elif 'Dilog_Act' in df.columns:
            dialogue_act_col = 'Dilog_Act'
        elif 'Unnamed: 3' in df.columns:
            dialogue_act_col = 'Unnamed: 3'
        else:
            raise ValueError(f'Undefined dialogue act column, file path \'{dialogue_file_path}\'.')
        # Pre-processed dialogue
        dialogue: Dict[str, Union[str, Dict[str, str]]] = {
            'split': self.split,
            'corpus': 'HOPE',
            'utterances': [
                {
                    'text': self._preprocess_text(row['Utterance']),
                    'speaker': self.SPEAKER_DECODER.get(row['Type']),
                    'dialogue_act': self.FINE_DIALOGUE_ACT_DECODER.get(
                        self.ACTION_REGEX.search(str(row[dialogue_act_col])).group(1).strip()
                    ),
                    'dialogue_act_category': self.COARSE_DIALOGUE_ACT_DECODER.get(
                        self.ACTION_REGEX.search(str(row[dialogue_act_col])).group(1).strip()
                    )
                }
                for _, row in df.iterrows()
            ]
        }

        return dialogue

    def _load_samples(self) -> List[Dict[str, Union[str, Dict[str, str]]]]:
        # Get sub-directory of current split
        if self.split == 'train':
            sub_dir_name = 'Train'
        elif self.split == 'validation':
            sub_dir_name = 'Validation'
        elif self.split == 'test':
            sub_dir_name = 'Test'
        else:
            raise ValueError(f'Unknown value for data set split: {self.split}')
        split_dir_path = os.path.join(self.corpus_dir_path, sub_dir_name)
        # Select file paths from current split
        dialogue_file_paths = [
            os.path.join(split_dir_path, transcripts_file)
            for transcripts_file in os.listdir(split_dir_path) if self.FILE_NAME_REGEX.match(transcripts_file)
        ]
        # Apply subsampling if required
        if self.sample is not None:
            # Get number of samples to collect
            if isinstance(self.sample, int):
                n_samples = self.sample
            else:
                n_samples = int(math.ceil(self.sample * len(dialogue_file_paths)))
            # Subsample data set unless the number of samples to take is equal to the number of samples available
            if n_samples != len(dialogue_file_paths):
                dialogue_file_paths = random.sample(dialogue_file_paths, n_samples)
        # Standardise corpus
        with parallel_backend(self.joblib_backend, n_jobs=self.n_jobs):
            dialogues = Parallel(verbose=self.verbosity_level)(
                delayed(self._preprocess_dialogue)(dialogue_file_path) for dialogue_file_path in dialogue_file_paths
            )

        return [dialogue for dialogue in dialogues if dialogue is not None]

    @classmethod
    def _get_labels(
            cls,
            dialogue: Dict,
            augmentation: bool,
            global_labels_metadata: Optional[Dict[str, str]],
            line_labels_metadata: Optional[Dict[str, str]]
    ) -> Tuple[Optional[Dict[str, str]], Optional[List[Dict[str, str]]]]:
        global_labels = None
        if line_labels_metadata is not None and len(line_labels_metadata) > 0:
            line_labels = [
                {
                    'dialogue_act': utterance['dialogue_act'],
                    'dialogue_act_category': utterance['dialogue_act_category']
                }
                for utterance in dialogue['utterances']
            ]
        else:
            line_labels = None

        return global_labels, line_labels

    @classmethod
    def _get_speakers(
            cls, dialogue: Dict, augmentation: bool
    ) -> Tuple[Tuple[str, str], bool]:
        therapist_fist = dialogue['utterances'][0]['speaker'] == 'therapist'
        if augmentation:
            bot = random.choice([True, False])
            if bot:
                if random.choice([True, False]):
                    sp1, sp2 = random.choice(
                        cls.HUMAN_THERAPY_BOT_SPEAKER_IDS[random.choice(list(cls.HUMAN_BOT_SPEAKER_IDS))])
                else:
                    sp1, sp2 = random.choice(
                        cls.HUMAN_PATIENT_BOT_SPEAKER_IDS[random.choice(list(cls.HUMAN_BOT_SPEAKER_IDS))])
            else:
                sp1, sp2 = random.choice(cls.HUMAN_SPEAKER_IDS[random.choice(list(cls.HUMAN_SPEAKER_IDS))])
        else:
            sp1, sp2 = cls.HUMAN_SPEAKER_IDS['default'][0]
            bot = False

        speakers = (sp1, sp2) if therapist_fist else (sp2, sp1)

        return speakers, bot

    @classmethod
    def _compose_premise(cls, speakers: Tuple[str, str], interaction: str, augmentation: bool, bot: bool) -> str:
        # Presentation line
        presentation = random.choice(cls.PRESENTATION) if augmentation else cls.PRESENTATION[0]
        # Generate speakers introduction and description
        if speakers[0].lower() in {'patient', 'user', 'client'} or any(role in speakers[0].lower() for role in {'patient', 'user', 'client'}):
            spp, spt = speakers
            speaker_first = False
        elif speakers[1].lower() in {'patient', 'user', 'client'} or any(role in speakers[1].lower() for role in {'patient', 'user', 'client'}):
            spt, spp = speakers
            speaker_first = True
        elif speakers[0].lower() in {'therapist', 'counsellor'} or any(role in speakers[0].lower() for role in {'therapist', 'counsellor'}):
            spt, spp = speakers
            speaker_first = True
        else:
            spp, spt = speakers
            speaker_first = False
        if bot:
            if any(spt.lower() in key.lower() or key.lower() in spt.lower() for key in
                   cls.HUMAN_THERAPY_BOT_SPEAKER_IDS):
                pt_list, pp_list = cls.THERAPY_BOT_PARTICIPANT, cls.PATIENT_HUMAN_PARTICIPANT
            else:
                pt_list, pp_list = cls.THERAPIST_HUMAN_PARTICIPANT, cls.PATIENT_BOT_PARTICIPANT
        else:
            pt_list, pp_list = cls.THERAPIST_HUMAN_PARTICIPANT, cls.PATIENT_HUMAN_PARTICIPANT
        pt = random.choice(pt_list) if augmentation else pt_list[0]
        pp = random.choice(pp_list) if augmentation else pp_list[0]
        if speaker_first:
            speakers_description = f'{pt}, {presentation} {spt}, and {pp}, {presentation} {spp}'
        else:
            speakers_description = f'{pp}, {presentation} {spp}, and {pt}, {presentation} {spt}'

        # Compose premise
        premise = random.choice(cls.PREMISE) if augmentation else cls.PREMISE[0]
        premise = premise.replace(INTERACTION, interaction)
        premise = premise.replace(SPEAKER_DESCRIPTIONS, speakers_description)
        premise = premise.replace(THERAPIST, spt)
        premise = premise.replace(PATIENT, spp)

        return premise

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
        if speakers[0].lower() in {'patient', 'user', 'client'} or any(role in speakers[0].lower() for role in {'patient', 'user', 'client'}):
            spp, spt = speakers
        elif speakers[1].lower() in {'patient', 'user', 'client'} or any(role in speakers[1].lower() for role in {'patient', 'user', 'client'}):
            spt, spp = speakers
        elif speakers[0].lower() in {'therapist', 'counsellor'} or any(role in speakers[0].lower() for role in {'therapist', 'counsellor'}):
            spt, spp = speakers
        else:
            spp, spt = speakers
        task_description = super(HOPE, cls)._compose_task_description(
            speakers, model_type, interaction, augmentation, dropout, bot,
            global_labels_metadata=global_labels_metadata, line_labels_metadata=line_labels_metadata,
            tgt_labels_metadata=tgt_labels_metadata, label_type=label_type, chunk=chunk, context=context,
            response=response
        )
        task_description = task_description.replace(THERAPIST, spt)
        task_description = task_description.replace(PATIENT, spp)

        return task_description


class EPITOME(_DialogueCorpus):
    # Corpus metadata
    IDENTIFIER = 'Empathy-Mental-Health-master'
    LINE_LABELS_METADATA: Dict[str, Dict[str, Optional[str]]] = {
        'emotional_reaction': {
            'id': 'emotional reaction',
            'description': 'Emotional reaction is a communication mechanism of empathy. '
                           'Having an emotional reaction means expressing emotions such as warmth, '
                           'compassion, and concern. Expressing these emotions plays an important role '
                           'in establishing empathic rapport and support',
            'values': ['no communication', 'weak communication', 'strong communication'],
            'explained': True
        },
        'exploration': {
            'id': 'exploration',
            'description': 'Exploration is a communication mechanism of empathy.'
                           'It consists in improving understanding of the other by exploring the feelings '
                           'and experiences not directly stated. Showing an active interest in what the other '
                           'is experiencing and feeling and probing gently is another important aspect of empathy',
            'values': ['no communication', 'weak communication', 'strong communication'],
            'explained': True
        },
        'interpretation': {
            'id': 'interpretation',
            'description': 'Interpretation is a communication mechanism of empathy. '
                           'Interpretation consists in communicating an understanding of feelings and experiences '
                           'inferred from the interaction with the other. Such a cognitive understanding in responses '
                           'is helpful in increasing awareness of hidden feelings and experiences, '
                           'and essential for developing alliance between two interacting',
            'values': ['no communication', 'weak communication', 'strong communication'],
            'explained': True
        }
    }
    # Composable strings to generate the dialogue
    INTERACTION_TYPE: List[str] = ['conversation', 'dialogue', 'chit-chat']
    CHUNK_TYPE: List[str] = ['window', 'chunk', 'piece', 'passage']
    CONTEXT_TYPE: List[str] = ['context', 'post']
    RESPONSE_TYPE: List[str] = ['response', 'utterance']
    # Premise
    PREMISE: List[str] = [
        f'The following is an exchange of messages on an online mental health support forum between {SPEAKER_DESCRIPTIONS}.',
        f'Here follows an exchange of messages on an online therapy forum between {SPEAKER_DESCRIPTIONS}.',
        f'This is an exchange of messages on an online mental health support forum between {SPEAKER_DESCRIPTIONS}.',
        f'The following is an exchange of messages on an online mental health support forum between {SPEAKER_DESCRIPTIONS}.',
        f'Here follows an exchange of messages on an online therapy forum between {SPEAKER_DESCRIPTIONS}.',
        f'This is an exchange of messages on an online mental health support forum between {SPEAKER_DESCRIPTIONS}.'
    ]
    THERAPIST_HUMAN_PARTICIPANT: List[str] = ['a therapist', 'a counsellor', 'a mental health professional', 'an empathetic listener']
    PATIENT_HUMAN_PARTICIPANT: List[str] = ['a person', 'a client']
    THERAPY_BOT_PARTICIPANT: List[str] = [
        'an empathetic AI', 'an empahtetic chatbot', 'an empathetic dialogue agent'
        'an AI for therapy', 'a chatbot for therapy', 'a conversational agent for therapy',
        'an dialogue agent for therapy'
        'an AI simulating a therapist', 'an AI', 'an AI system simulating a therapist',
        'an AI system', 'a chatbot simulating a therapist', 'a chatbot',
        'a conversational agent simulating a therapist', 'a conversational agent',
        'an dialogue agent simulating a therapist', 'an dialogue agent'
    ]
    PATIENT_BOT_PARTICIPANT: List[str] = [
        'an AI simulating a support seeking person', 'an AI', 'an AI system simulating a support seeking person',
        'an AI system', 'a chatbot simulating a support seeking person', 'a chatbot',
        'a conversational agent simulating a support seeking person', 'a conversational agent',
        'a dialogue agent simulating a support seeking person', 'an dialogue agent'
    ]
    DISCRIMINATIVE_TASK_PROMPT: Dict[str, List[str]] = {
        'local': [f'{LABEL_ID}{CLS_SEP}', f'{RESPONSE} {LABEL_ID}{CLS_SEP}', f'The {LABEL_ID} of the {RESPONSE} is']
    }
    GENERATIVE_TASK_PREFIX: List[str] = [
        f'In the following interactions, {SPEAKERS} will converse in natural language. '
        f'The {PATIENT} talks about his/hers issues to the {THERAPIST} '
        f'and the {THERAPIST} offers support to the {PATIENT}.',
        f'During these interactions, {SPEAKERS} will converse in natural language. '
        f'The {PATIENT} talks about his/hers issues to the {THERAPIST} '
        f'and the {THERAPIST} offers support to the {PATIENT}.',
        f'In the following interactions, {SPEAKERS} converse in natural language. '
        f'The {PATIENT} talks about his/hers issues to the {THERAPIST} '
        f'and the {THERAPIST} offers support to the {PATIENT}.',
        f'During these interactions, {SPEAKERS} converse in natural language. '
        f'The {PATIENT} talks about his/hers issues to the {THERAPIST} '
        f'and the {THERAPIST} offers support to the {PATIENT}.',
        f'In the following interactions, {SPEAKERS} are conversing in natural language. '
        f'The {PATIENT} talks about his/hers issues to the {THERAPIST} '
        f'and the {THERAPIST} offers support to the {PATIENT}.',
        f'During these interactions, {SPEAKERS} are conversing in natural language. '
        f'The {PATIENT} talks about his/hers issues to the {THERAPIST} '
        f'and the {THERAPIST} offers support to the {PATIENT}.'
    ]
    EXTENDED_GENERATIVE_TASK_DESCRIPTION: Dict[bool, str] = {
        False: f'Whenever the considered aspect of empathy is present (i.e., if the communication is either '
               f'weak or strong), explain the rationale behind it by reporting the passage(s) of the {RESPONSE} '
               f'motivating the answer.',
        True: f'Whenever one the considered aspects of empathy are present (i.e., if the communication is either '
              f'weak or strong), explain the rationale behind it by reporting the passage(s) of the {RESPONSE} '
              f'motivating the answer.'
    }
    EXPLANATION_PROMPT: List[str] = ['. Rationale: ', ', rationale: ', '. Passage(s): ', ', passage(s): ']
    P_DROP_EXPLANATION = 0.3
    # IDs
    HUMAN_SPEAKER_IDS: Dict[str, List[Tuple[str, str]]] = {
        'default': [
            ('Supporter', 'User'),
            ('Supporter', 'Client'),
            ('Supporter', 'Person')
        ]
    }
    HUMAN_THERAPY_BOT_SPEAKER_IDS: Dict[str, List[Tuple[str, str]]] = {
        'AI': [
            ('AI', 'Person'),
            ('AI', 'Client'),
            ('AI', 'User'),
            ('Therapy AI', 'Person'),
            ('Therapy AI', 'Client'),
            ('Therapy AI', 'User'),
            ('Support AI', 'Person'),
            ('Support AI', 'Client'),
            ('Support AI', 'User'),
        ],
        'chatbot': [
            ('Chatbot', 'Person'),
            ('Chatbot', 'Client'),
            ('Chatbot', 'User'),
            ('Therapy Chatbot', 'Person'),
            ('Therapy Chatbot', 'Client'),
            ('Therapy Chatbot', 'User'),
            ('Support Chatbot', 'Person'),
            ('Support Chatbot', 'Client'),
            ('Support Chatbot', 'User')
        ]
    }
    HUMAN_PATIENT_BOT_SPEAKER_IDS: Dict[str, List[Tuple[str, str]]] = {
        'AI': [
            ('Therapist', 'AI'),
            ('Counsellor', 'AI'),
            ('Therapist', 'Client AI'),
            ('Counsellor', 'Client AI'),
            ('Therapist', 'Patient AI'),
            ('Counsellor', 'Patient AI')
        ],
        'chatbot': [
            ('Therapist', 'Chatbot'),
            ('Counsellor', 'Chatbot'),
            ('Therapist', 'Client Chatbot'),
            ('Counsellor', 'Client Chatbot'),
            ('Therapist', 'Patient Chatbot'),
            ('Counsellor', 'Patient Chatbot')
        ],
    }
    # Data loading and preprocessing parameters
    FILE_LIST: Dict[str, str] = {
        'emotional-reaction': 'emotional-reactions-reddit.csv',
        'exploration': 'explorations-reddit.csv',
        'interpretation': 'interpretations-reddit.csv'
    }
    LEVEL_DECODER = ['no communication', 'weak communication', 'strong communication']


    def _preprocess_dialogue(self, dialogue_df_row: pd.Series) -> Dict[str, Union[str, Dict[str, str]]]:
        # Pre-processed dialogue
        dialogue: Dict[str, Union[str, Dict[str, str]]] = {
            'split': self.split,
            'corpus': 'EPITOME',
            'utterances': [
                {'text': self._preprocess_text(dialogue_df_row['seeker_post']), 'speaker': 'patient'},
                {
                    'speaker': 'therapist',
                    'text': self._preprocess_text(dialogue_df_row['response_post']),
                    'emotional_reaction': self.LEVEL_DECODER[dialogue_df_row['level_er']],
                    'rationale_emotional_reaction': ', '.join(
                        [f'"{s}"' if s != 'N.A' else s for s in dialogue_df_row['rationales_er'].split('|') if s != '']
                    ),
                    'exploration': self.LEVEL_DECODER[dialogue_df_row['level_e']],
                    'rationale_exploration': ', '.join(
                        [f'"{s}"' if s != 'N.A' else s for s in dialogue_df_row['rationales_e'].split('|') if s != '']
                    ),
                    'interpretation': self.LEVEL_DECODER[dialogue_df_row['level_i']],
                    'rationale_interpretation': ', '.join(
                        [f'"{s}"' if s != 'N.A' else s for s in dialogue_df_row['rationales_i'].split('|') if s != '']
                    )
                }
            ]
        }

        return dialogue

    def _load_samples(self) -> List[Dict[str, Union[str, Dict[str, str]]]]:
        # Load CSV files
        df_er = pd.read_csv(os.path.join(self.corpus_dir_path, 'dataset', self.FILE_LIST['emotional-reaction']))
        df_e = pd.read_csv(os.path.join(self.corpus_dir_path, 'dataset', self.FILE_LIST['exploration']))
        df_i = pd.read_csv(os.path.join(self.corpus_dir_path, 'dataset', self.FILE_LIST['interpretation']))
        # Join data frames
        df = df_er[['sp_id', 'rp_id', 'seeker_post', 'response_post']]
        df[['level_er', 'rationales_er']] = df_er[['level', 'rationales']]
        df[['level_e', 'rationales_e']] = df_e[['level', 'rationales']]
        df[['level_i', 'rationales_i']] = df_i[['level', 'rationales']]
        # Clear NaNs
        df[['rationales_er', 'rationales_e', 'rationales_i']] = df[['rationales_er', 'rationales_e', 'rationales_i']].fillna('N.A')
        # Do train-validation-test split on the indices
        train_df, test_df = train_test_split(df, test_size=self.holdout, random_state=self.random_seed)
        train_df, validation_df = train_test_split(train_df, test_size=self.holdout, random_state=self.random_seed)
        # Get list of current split indices
        if self.split == 'train':
            df = train_df
        elif self.split == 'validation':
            df = validation_df
        elif self.split == 'test':
            df = test_df
        else:
            raise ValueError(f'Unknown value for data set split: {self.split}')
        # Apply subsampling if required
        if self.sample is not None:
            # Get number of samples to collect
            if isinstance(self.sample, int):
                n_samples = self.sample
            else:
                n_samples = int(math.ceil(self.sample * len(df)))
            # Subsample data set unless the number of samples to take is equal to the number of samples available
            if n_samples != len(df):
                df = df.sample(n_samples)
        # Standardise corpus
        with parallel_backend(self.joblib_backend, n_jobs=self.n_jobs):
            return Parallel(verbose=self.verbosity_level)(
                delayed(self._preprocess_dialogue)(row) for _, row in df.iterrows()
            )

    @classmethod
    def _get_labels(
            cls,
            dialogue: Dict,
            augmentation: bool,
            global_labels_metadata: Optional[Dict[str, str]],
            line_labels_metadata: Optional[Dict[str, str]]
    ) -> Tuple[Optional[Dict[str, str]], Optional[List[Dict[str, str]]]]:
        global_labels = None
        if line_labels_metadata is not None and len(line_labels_metadata) > 0:
            line_labels = [
                None,
                {
                    'emotional_reaction': dialogue['utterances'][1]['emotional_reaction'],
                    'rationale_emotional_reaction': dialogue['utterances'][1]['rationale_emotional_reaction'],
                    'exploration': dialogue['utterances'][1]['exploration'],
                    'rationale_exploration': dialogue['utterances'][1]['rationale_exploration'],
                    'interpretation': dialogue['utterances'][1]['interpretation'],
                    'rationale_interpretation': dialogue['utterances'][1]['rationale_interpretation'],
                }
            ]
        else:
            line_labels = None

        return global_labels, line_labels

    @classmethod
    def _get_speakers(
            cls, dialogue: Dict, augmentation: bool
    ) -> Tuple[Tuple[str, str], bool]:
        therapist_fist = dialogue['utterances'][0]['speaker'] == 'therapist'
        if augmentation:
            bot = random.choice([True, False])
            if bot:
                if random.choice([True, False]):
                    sp1, sp2 = random.choice(
                        cls.HUMAN_THERAPY_BOT_SPEAKER_IDS[random.choice(list(cls.HUMAN_BOT_SPEAKER_IDS))])
                else:
                    sp1, sp2 = random.choice(
                        cls.HUMAN_PATIENT_BOT_SPEAKER_IDS[random.choice(list(cls.HUMAN_BOT_SPEAKER_IDS))])
            else:
                sp1, sp2 = random.choice(cls.HUMAN_SPEAKER_IDS[random.choice(list(cls.HUMAN_SPEAKER_IDS))])
        else:
            sp1, sp2 = cls.HUMAN_SPEAKER_IDS['default'][0]
            bot = False

        speakers = (sp1, sp2) if therapist_fist else (sp2, sp1)

        return speakers, bot

    @classmethod
    def _prepare_labels_metadata(cls, labels_metadata: Dict, augmentation: bool, dropout: List[bool]) -> Dict:
        labels_metadata = super(EPITOME, cls)._prepare_labels_metadata(labels_metadata, augmentation, dropout)
        if any(dropout) and random.uniform(0.0, 1.0) < cls.P_DROP_EXPLANATION:
            for lbl in labels_metadata:
                labels_metadata[lbl]['explained'] = False

        return labels_metadata

    @classmethod
    def _compose_premise(cls, speakers: Tuple[str, str], interaction: str, augmentation: bool, bot: bool) -> str:
        # Presentation line
        presentation = random.choice(cls.PRESENTATION) if augmentation else cls.PRESENTATION[0]
        # Generate speakers introduction and description
        if speakers[0].lower() in {'patient', 'user', 'client'} or any(role in speakers[0].lower() for role in {'patient', 'user', 'client'}):
            spp, spt = speakers
            speaker_first = False
        elif speakers[1].lower() in {'patient', 'user', 'client'} or any(role in speakers[1].lower() for role in {'patient', 'user', 'client'}):
            spt, spp = speakers
            speaker_first = True
        elif speakers[0].lower() in {'therapist', 'counsellor'} or any(role in speakers[0].lower() for role in {'therapist', 'counsellor'}):
            spt, spp = speakers
            speaker_first = True
        else:
            spp, spt = speakers
            speaker_first = False
        if bot:
            if any(spt.lower() in key.lower() or key.lower() in spt.lower() for key in
                   cls.HUMAN_THERAPY_BOT_SPEAKER_IDS):
                pt_list, pp_list = cls.THERAPY_BOT_PARTICIPANT, cls.PATIENT_HUMAN_PARTICIPANT
            else:
                pt_list, pp_list = cls.THERAPIST_HUMAN_PARTICIPANT, cls.PATIENT_BOT_PARTICIPANT
        else:
            pt_list, pp_list = cls.THERAPIST_HUMAN_PARTICIPANT, cls.PATIENT_HUMAN_PARTICIPANT
        pt = random.choice(pt_list) if augmentation else pt_list[0]
        pp = random.choice(pp_list) if augmentation else pp_list[0]
        if speaker_first:
            speakers_description = f'{pt}, {presentation} {spt}, and {pp}, {presentation} {spp}'
        else:
            speakers_description = f'{pp}, {presentation} {spp}, and {pt}, {presentation} {spt}'

        # Compose premise
        premise = random.choice(cls.PREMISE) if augmentation else cls.PREMISE[0]
        premise = premise.replace(INTERACTION, interaction)
        premise = premise.replace(SPEAKER_DESCRIPTIONS, speakers_description)
        premise = premise.replace(THERAPIST, spt)
        premise = premise.replace(PATIENT, spp)

        return premise

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
        if speakers[0].lower() in {'patient', 'user', 'client'} or any(role in speakers[0].lower() for role in {'patient', 'user', 'client'}):
            spp, spt = speakers
        elif speakers[1].lower() in {'patient', 'user', 'client'} or any(role in speakers[1].lower() for role in {'patient', 'user', 'client'}):
            spt, spp = speakers
        elif speakers[0].lower() in {'therapist', 'counsellor'} or any(role in speakers[0].lower() for role in {'therapist', 'counsellor'}):
            spt, spp = speakers
        else:
            spp, spt = speakers
        task_description = super(EPITOME, cls)._compose_task_description(
            speakers, model_type, interaction, augmentation, dropout, bot,
            global_labels_metadata=global_labels_metadata, line_labels_metadata=line_labels_metadata,
            tgt_labels_metadata=tgt_labels_metadata, label_type=label_type, chunk=chunk, context=context,
            response=response
        )
        task_description = task_description.replace(THERAPIST, spt)
        task_description = task_description.replace(PATIENT, spp)

        if model_type == 'discriminator' and tgt_labels_metadata is not None and len(tgt_labels_metadata) > 0 and all(tgt_labels_metadata[lbl]['explained'] for lbl in tgt_labels_metadata):
            task_description += '\n' + cls.EXTENDED_GENERATIVE_TASK_DESCRIPTION[len(tgt_labels_metadata) > 1].replace(RESPONSE, response)

        return task_description

    @classmethod
    def _get_dialogue_labels(cls, *args, **kwargs) -> List[str]:
        tmp_dialogue_labels = super(EPITOME, cls)._get_dialogue_labels(*args, **kwargs)
        # Add explanations if required
        tgt_labels_metadata = args[1]
        if all(tgt_labels_metadata[lbl]['explained'] for lbl in tgt_labels_metadata):
            dialogue_labels = []
            expl_sep = random.choice(cls.EXPLANATION_PROMPT) if args[2] else cls.EXPLANATION_PROMPT[0]
            for lbl, lbl_line in zip(tgt_labels_metadata, tmp_dialogue_labels):
                eol = lbl_line[-2:] if lbl_line.endswith(' ') or lbl_line.endswith('\n') else lbl_line[-1:]
                lbl_line = f'{lbl_line[:-len(eol)]}{expl_sep}{args[0][f"rationale_{lbl}"]}{eol}'
                dialogue_labels.append(lbl_line)
        else:
            dialogue_labels = tmp_dialogue_labels

        return dialogue_labels

    @classmethod
    def _compose_discriminator_task(cls, *args, **kwargs) -> str:
        try:
            return super(EPITOME, cls)._compose_discriminator_task(*args, **kwargs)
        except TypeError:
            return NULL

    @classmethod
    def _compose_discriminator_dialogue(cls, *args, **kwargs) -> List[str]:
        # Drop additional dummy turn
        return super(EPITOME, cls)._compose_discriminator_dialogue(*args, **kwargs)[1:]

    @classmethod
    def _compose_discriminator_dialogue_eval(cls, *args, **kwargs):
        line_labels_metadata = deepcopy(kwargs['line_labels_metadata'])
        for lbl in line_labels_metadata:
            line_labels_metadata[lbl]['explained'] = False
        kwargs['line_labels_metadata'] = line_labels_metadata
        seqs = super(EPITOME, cls)._compose_discriminator_dialogue_eval(*args, **kwargs)
        approach = args[2]
        if approach == 'posterior':
            seqs['passages'] = seqs['passages'][1:]
        elif approach == 'infilling' or approach == 'prediction':
            seqs['utterances'] = seqs['utterances'][1:]
        else:
            raise ValueError(f'Unknown approach requested \'{approach}\'')

        return seqs

    @classmethod
    def _compose_explanation_dialogue_eval(cls, *args, **kwargs):
        line_labels_metadata = deepcopy(kwargs['line_labels_metadata'])
        for lbl in line_labels_metadata:
            line_labels_metadata[lbl]['explained'] = False
        kwargs['line_labels_metadata'] = line_labels_metadata
        seqs = super(EPITOME, cls)._compose_discriminator_dialogue_eval(*args, **kwargs)
        seqs['passages'] = seqs['passages'][1:]
        tokeniser = args[3]
        # Extend annotation or generation
        annotations = seqs['passages'][0].pop('annotations')
        target = seqs['passages'][0].pop('target')
        # Get index of correct answer
        y_true = [target in annotation for annotation in annotations].index(True)
        # Add target string
        seqs['passages'][0]['explanation'] = annotations[y_true].replace(
            '.' + tokeniser.eos_token,
            f'{cls.EXPLANATION_PROMPT[0]}{kwargs["line_labels"][-1]["rationale_{}".format(list(kwargs["line_labels_metadata"].keys())[0])]}.'
            f'{tokeniser.eos_token}'
        )

        return seqs

    @classmethod
    def _compose_generator_dialogue_eval(cls, *args, **kwargs):
        seqs = super(EPITOME, cls)._compose_generator_dialogue_eval(*args, **kwargs)
        seqs['task_description'] += f'\n\n{"{} {}".format(*seqs["utterances"][0])}'
        seqs['utterances'] = seqs['utterances'][1:]

        return seqs


class CounselChat(_DialogueCorpus):
    IDENTIFIER: str = 'Counsel_Chat'
    # Composable strings to generate the utterances
    INTERACTION_TYPE: List[str] = ['conversation', 'utterances', 'chit-chat']
    CHUNK_TYPE: List[str] = ['window', 'chunk', 'piece', 'passage']
    CONTEXT_TYPE: List[str] = ['context', 'history', 'past']
    RESPONSE_TYPE: List[str] = ['response', 'utterance']
    # Premise
    PREMISE: List[str] = [
        f'The following is an exchange of messages on an online therapy forum between {SPEAKER_DESCRIPTIONS}.',
        f'Here follows an exchange of messages on an online therapy forum between {SPEAKER_DESCRIPTIONS}.',
        f'This is an exchange of messages on an online therapy forum between {SPEAKER_DESCRIPTIONS}.',
        f'The following is an exchange of messages on an online counselling forum between {SPEAKER_DESCRIPTIONS}.',
        f'Here follows an exchange of messages on an online counselling forum between {SPEAKER_DESCRIPTIONS}.',
        f'This is an exchange of messages on an online counselling forum between {SPEAKER_DESCRIPTIONS}.'
    ]
    THERAPIST_HUMAN_PARTICIPANT: List[str] = ['a therapist', 'a counsellor', 'a mental health professional']
    PATIENT_HUMAN_PARTICIPANT: List[str] = ['a person', 'a client']
    THERAPY_BOT_PARTICIPANT: List[str] = [
        'an AI for therapy', 'a chatbot for therapy', 'a conversational agent for therapy',
        'a dialogue agent for therapy'
        'an AI simulating a therapist', 'an AI', 'an AI system simulating a therapist',
        'an AI system', 'a chatbot simulating a therapist', 'a chatbot',
        'a conversational agent simulating a therapist', 'a conversational agent',
        'an dialogue agent simulating a therapist', 'an dialogue agent'
    ]
    PATIENT_BOT_PARTICIPANT: List[str] = [
        'an AI simulating a therapy patient', 'an AI', 'an AI system simulating a therapy patient',
        'an AI system', 'a chatbot simulating a therapy patient', 'a chatbot',
        'a conversational agent simulating a therapy patient', 'a conversational agent',
        'a dialogue agent simulating a therapy patient', 'a dialogue agent'
    ]
    DISCRIMINATIVE_TASK_PROMPT: Dict[str, List[str]] = {
        'global': [
            f'{LABEL_ID}{CLS_SEP}',
            f'{LABEL_ID} in the {INTERACTION} {DIALOGUE_CHUNK} {CLS_SEP}',
            f'The {LABEL_ID} in the {INTERACTION} {DIALOGUE_CHUNK} is'
        ],
        'local': [f'{LABEL_ID}{CLS_SEP}', f'{RESPONSE} {LABEL_ID}{CLS_SEP}', f'The {LABEL_ID} of the {RESPONSE} is']
    }
    GENERATIVE_TASK_PREFIX: List[str] = [
        f'In the following interactions, {SPEAKERS} will converse in natural language. '
        f'The {PATIENT} talks about his/hers issues to the {THERAPIST} '
        f'and the {THERAPIST} helps the {PATIENT} explore and solve his/hers problems.',
        f'During these interactions, {SPEAKERS} will converse in natural language. '
        f'The {PATIENT} talks about his/hers issues to the {THERAPIST} '
        f'and the {THERAPIST} helps the {PATIENT} explore and solve his/hers problems.',
        f'In the following interactions, {SPEAKERS} converse in natural language. '
        f'The {PATIENT} talks about his/hers issues to the {THERAPIST} '
        f'and the {THERAPIST} helps the {PATIENT} explore and solve his/hers problems.',
        f'During these interactions, {SPEAKERS} converse in natural language. '
        f'The {PATIENT} talks about his/hers issues to the {THERAPIST} '
        f'and the {THERAPIST} helps the {PATIENT} explore and solve his/hers problems.',
        f'In the following interactions, {SPEAKERS} are conversing in natural language. '
        f'The {PATIENT} talks about his/hers issues to the {THERAPIST} '
        f'and the {THERAPIST} helps the {PATIENT} explore and solve his/hers problems.',
        f'During these interactions, {SPEAKERS} are conversing in natural language. '
        f'The {PATIENT} talks about his/hers issues to the {THERAPIST} '
        f'and the {THERAPIST} helps the {PATIENT} explore and solve his/hers problems.'
    ]
    # IDs
    HUMAN_SPEAKER_IDS: Dict[str, List[Tuple[str, str]]] = {
        'default': [
            ('Therapist', 'Patient'),
            ('Counsellor', 'Patient'),
            ('Therapist', 'Client'),
            ('Counsellor', 'Client')
        ]
    }
    HUMAN_THERAPY_BOT_SPEAKER_IDS: Dict[str, List[Tuple[str, str]]] = {
        'AI': [
            ('AI', 'Patient'),
            ('AI', 'Client'),
            ('AI', 'User'),
            ('Therapy AI', 'Patient'),
            ('Therapy AI', 'Client'),
            ('Therapy AI', 'User'),
            ('Counselling AI', 'Patient'),
            ('Counselling AI', 'Client'),
            ('Counselling AI', 'User')
        ],
        'chatbot': [
            ('Chatbot', 'Patient'),
            ('Chatbot', 'Client'),
            ('Chatbot', 'User'),
            ('Therapy Chatbot', 'Patient'),
            ('Therapy Chatbot', 'Client'),
            ('Therapy Chatbot', 'User'),
            ('Counselling Chatbot', 'Patient'),
            ('Counselling Chatbot', 'Client'),
            ('Counselling Chatbot', 'User')
        ]
    }
    HUMAN_PATIENT_BOT_SPEAKER_IDS: Dict[str, List[Tuple[str, str]]] = {
        'AI': [
            ('Therapist', 'AI'),
            ('Counsellor', 'AI'),
            ('Therapist', 'Client AI'),
            ('Counsellor', 'Client AI'),
            ('Therapist', 'Patient AI'),
            ('Counsellor', 'Patient AI')
        ],
        'chatbot': [
            ('Therapist', 'Chatbot'),
            ('Counsellor', 'Chatbot'),
            ('Therapist', 'Client Chatbot'),
            ('Counsellor', 'Client Chatbot'),
            ('Therapist', 'Patient Chatbot'),
            ('Counsellor', 'Patient Chatbot')
        ],
    }

    def _preprocess_dialogue(self, dialogue: Dict) -> Dict[str, Union[str, Dict[str, str]]]:
        # Corpus metadata
        dialogue = {
            'split': self.split,
            'corpus': 'CounselChat',
            'utterances': [
                {'text': dialogue['utterances'][-1]['history'][-1], 'speaker': 'patient'},
                {'text': dialogue['utterances'][-1]['candidates'][-1], 'speaker': 'therapist'}
            ]
        }

        return dialogue

    def _load_samples(self) -> List[Dict[str, Union[str, Dict[str, str]]]]:
        # NOTE the data set does not provide a test split so we will use the validation split as test and we will adda custom validation split
        # Read JSON file
        with open(os.path.join(self.corpus_dir_path, 'counselchatdata.json')) as f:
            dialogues = json.load(f)
        # Get current split data
        if self.split == 'train' or self.split == 'validation':
            dialogues = dialogues['train']
            # Get indices list
            idxs = range(len(dialogues))
            # Do train-validation-test split on the indices
            train_idxs, validation_idxs = train_test_split(idxs, test_size=self.holdout, random_state=self.random_seed)
            if self.split == 'train':
                idxs = train_idxs
            elif self.split == 'validation':
                idxs = validation_idxs
            # Select samples from current split
            dialogues = [dialogues[i] for i in idxs]
        elif self.split == 'test':
            dialogues = dialogues['valid']
        else:
            raise ValueError(f'Unknown value for data set split: {self.split}')
        # Apply subsampling if required
        if self.sample is not None:
            # Get number of samples to collect
            n_samples = self.sample if isinstance(self.sample, int) else int(
                math.ceil(self.sample * len(dialogues)))
            # Subsample data set unless the number of samples to take is equal to the number of samples available
            if n_samples != len(dialogues):
                dialogues = random.sample(dialogues, n_samples)
        # Standardise corpus
        with parallel_backend(self.joblib_backend, n_jobs=self.n_jobs):
            return Parallel(verbose=self.verbosity_level)(
                delayed(self._preprocess_dialogue)(dialogue) for dialogue in dialogues
            )

    @classmethod
    def _get_speakers(
            cls, dialogue: Dict, augmentation: bool
    ) -> Tuple[Tuple[str, str], bool]:
        therapist_fist = dialogue['utterances'][0]['speaker'] == 'therapist'
        if augmentation:
            bot = random.choice([True, False])
            if bot:
                if random.choice([True, False]):
                    sp1, sp2 = random.choice(
                        cls.HUMAN_THERAPY_BOT_SPEAKER_IDS[random.choice(list(cls.HUMAN_BOT_SPEAKER_IDS))])
                else:
                    sp1, sp2 = random.choice(
                        cls.HUMAN_PATIENT_BOT_SPEAKER_IDS[random.choice(list(cls.HUMAN_BOT_SPEAKER_IDS))])
            else:
                sp1, sp2 = random.choice(cls.HUMAN_SPEAKER_IDS[random.choice(list(cls.HUMAN_SPEAKER_IDS))])
        else:
            sp1, sp2 = cls.HUMAN_SPEAKER_IDS['default'][0]
            bot = False

        speakers = (sp1, sp2) if therapist_fist else (sp2, sp1)

        return speakers, bot

    @classmethod
    def _compose_premise(cls, speakers: Tuple[str, str], interaction: str, augmentation: bool, bot: bool) -> str:
        # Presentation line
        presentation = random.choice(cls.PRESENTATION) if augmentation else cls.PRESENTATION[0]
        # Generate speakers introduction and description
        if speakers[0].lower() in {'patient', 'user', 'client'} or any(role in speakers[0].lower() for role in {'patient', 'user', 'client'}):
            spp, spt = speakers
            speaker_first = False
        elif speakers[1].lower() in {'patient', 'user', 'client'} or any(role in speakers[1].lower() for role in {'patient', 'user', 'client'}):
            spt, spp = speakers
            speaker_first = True
        elif speakers[0].lower() in {'therapist', 'counsellor'} or any(role in speakers[0].lower() for role in {'therapist', 'counsellor'}):
            spt, spp = speakers
            speaker_first = True
        else:
            spp, spt = speakers
            speaker_first = False
        if bot:
            if any(spt.lower() in key.lower() or key.lower() in spt.lower() for key in
                   cls.HUMAN_THERAPY_BOT_SPEAKER_IDS):
                pt_list, pp_list = cls.THERAPY_BOT_PARTICIPANT, cls.PATIENT_HUMAN_PARTICIPANT
            else:
                pt_list, pp_list = cls.THERAPIST_HUMAN_PARTICIPANT, cls.PATIENT_BOT_PARTICIPANT
        else:
            pt_list, pp_list = cls.THERAPIST_HUMAN_PARTICIPANT, cls.PATIENT_HUMAN_PARTICIPANT
        pt = random.choice(pt_list) if augmentation else pt_list[0]
        pp = random.choice(pp_list) if augmentation else pp_list[0]
        if speaker_first:
            speakers_description = f'{pt}, {presentation} {spt}, and {pp}, {presentation} {spp}'
        else:
            speakers_description = f'{pp}, {presentation} {spp}, and {pt}, {presentation} {spt}'

        # Compose premise
        premise = random.choice(cls.PREMISE) if augmentation else cls.PREMISE[0]
        premise = premise.replace(INTERACTION, interaction)
        premise = premise.replace(SPEAKER_DESCRIPTIONS, speakers_description)
        premise = premise.replace(THERAPIST, spt)
        premise = premise.replace(PATIENT, spp)

        return premise

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
        if speakers[0].lower() in {'patient', 'user', 'client'} or any(role in speakers[0].lower() for role in {'patient', 'user', 'client'}):
            spp, spt = speakers
        elif speakers[1].lower() in {'patient', 'user', 'client'} or any(role in speakers[1].lower() for role in {'patient', 'user', 'client'}):
            spt, spp = speakers
        elif speakers[0].lower() in {'therapist', 'counsellor'} or any(role in speakers[0].lower() for role in {'therapist', 'counsellor'}):
            spt, spp = speakers
        else:
            spp, spt = speakers
        task_description = super(CounselChat, cls)._compose_task_description(
            speakers, model_type, interaction, augmentation, dropout, bot,
            global_labels_metadata=global_labels_metadata, line_labels_metadata=line_labels_metadata,
            tgt_labels_metadata=tgt_labels_metadata, label_type=label_type, chunk=chunk, context=context,
            response=response
        )
        task_description = task_description.replace(THERAPIST, spt)
        task_description = task_description.replace(PATIENT, spp)

        return task_description