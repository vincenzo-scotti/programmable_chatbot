import numpy as np
from sklearn.metrics import classification_report
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from programmable_chatbot.utils import IGNORE_INDEX, nll

from typing import Optional, Union, List, Literal, Tuple, Dict


class Chatbot:
    def __init__(
            self,
            model: Union[str, GPT2LMHeadModel],
            tokenizer: Optional[Union[str, GPT2Tokenizer]] = None,
            max_response_length: Optional[int] = None,
            device: Optional[torch.device] = None,
            mixed_precision: bool = True,
            in_mem: Optional[int] = None
    ):
        self.device: torch.device = device if device is not None else (
            torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        )
        self.model = model if isinstance(model, GPT2LMHeadModel) else GPT2LMHeadModel.from_pretrained(model)
        self.model.to(device)
        self.tokenizer = GPT2Tokenizer.from_pretrained(tokenizer) if isinstance(tokenizer, str) else tokenizer
        self.max_response_length: Optional[int] = max_response_length
        self.mixed_precision: bool = mixed_precision
        self.in_mem: Optional[int] = in_mem

    def __call__(self, *args, **kwargs):
        ...

    def generate(self, context: str) -> str:
        ...

    def label_utterance(self):
        ...

    def label_dialogue(self):
        ...

    def _nll(
            self,
            logits,
            labels,
            reduction: Optional[Literal['mean', 'sum', 'batchmean', 'batchsum', 'seqmean', 'seqsum']] = None
    ) -> torch.tensor:
        return nll(logits, labels, reduction=reduction)

    def score(
            self,
            sequences: Union[str, List[str]],
            *args,
            past_attention_mask: Optional[torch.tensor] = None,
            nll_reduction: Optional[str] = None,
            return_model_output: bool = False,
            **kwargs,
    ):
        # If there is a single sequence, wrap it into a list and then unpack the output
        if isinstance(sequences, str):
            score_output = self.score(
                [sequences],
                *args,
                past_attention_mask=past_attention_mask,
                nll_reduction=nll_reduction,
                return_model_output=return_model_output,
                **kwargs
            )
            if return_model_output:
                return score_output[0][0], score_output[1]
            else:
                return score_output[0]
        #
        with torch.no_grad(), torch.autocast(self.device.type, enabled=self.mixed_precision):
            # Encode input sequences
            input_encodings = self.tokenizer(sequences, return_tensors='pt', padding=True).to(self.device)
            # IDs
            input_ids = input_encodings.input_ids
            # Attention mask
            attention_mask = input_encodings.attention_mask
            # Target labels (they will be shifted automatically by NLL computation)
            labels = input_ids.clone()
            labels[~attention_mask.bool()] = IGNORE_INDEX
            if attention_mask is not None:
                attention_mask = torch.hstack([past_attention_mask, attention_mask])
            # Get model predictions
            output = self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
            logits = output.logits
            # Compute token-wise loss
            nll = self._nll(logits, labels, reduction=nll_reduction)

            if return_model_output:
                return nll, output
            else:
                return nll

    def score_dialogue(self, task_description: str, utterances: List[Tuple[str, str]]) -> List[float]:
        # Encode task description
        task_description_encodings = self.tokenizer(f'{task_description}\n\n', return_tensors='pt').to(self.device)
        # Prepare accumulators
        attention_mask = task_description_encodings.attention_mask
        past_key_values = self.model.transformer(**task_description_encodings).past_key_values
        # Parameters
        task_description_len: int = task_description_encodings.input_ids.size(1)
        ellipsis_len: int = len(self.tokenizer('...\n\n').input_ids)
        utterances_len = []
        context_len = 0
        # Perplexity accumulator
        ppl = []
        # Iterate over utterances
        for idx, (prompt, utterance) in enumerate(utterances):
            # Encode current prompt-utterance pair
            utterance_input_ids, utterance_attention_mask = self.tokenizer(f'{prompt} {utterance}', return_tensors='pt').to(self.device).values()
            # Get current response length
            utterances_len.append(utterance_input_ids.size(0))
            # Get current prompt length
            prompt_length = len(self.tokenizer(prompt))
            # Check if maximum length was reached and reset past
            if attention_mask.size(0) + utterances_len[-1] > self.tokenizer.model_max_length:
                #
                context_idx = 0
                while context_idx < idx and context_len + ellipsis_len + task_description_len + utterances_len[-1] > self.tokenizer.model_max_length:
                    context_len -= utterances_len[context_idx]
                    context_idx += 1
                # Encode and process the new context
                context_encodings = self.tokenizer(
                    f'{task_description}\n\n...\n\n{str().join(f"{p} {u}" for p, u in utterances[context_idx:idx])}',
                    return_tensors='pt'
                ).to(self.device)
                # Update past
                attention_mask = context_encodings.attention_mask
                past_key_values = self.model.transformer(**context_encodings).past_key_values
            # Get maximum response length
            max_response_len = self.tokenizer.model_max_length - attention_mask.size(0)
            # Target labels (they will be shifted automatically by NLL computation)
            labels = utterance_input_ids[:, prompt_length:max_response_len].clone()
            labels[~utterance_attention_mask[:, prompt_length:max_response_len].bool()] = IGNORE_INDEX
            # Get model predictions
            print(utterance_input_ids.size(), attention_mask.size(), utterance_attention_mask.size(), past_key_values[0][0].size())
            output = self.model(
                input_ids=utterance_input_ids[:max_response_len],
                attention_mask=torch.hstack([attention_mask, utterance_attention_mask[:max_response_len]]),
                past_key_values=past_key_values
            )
            logits = output.logits[:, prompt_length:]
            # Compute sequence PPL
            ppl.append(self._nll(logits, labels, reduction='seqmean').squeeze().exp())
            # Update attention mask
            attention_mask = torch.hstack([attention_mask, utterance_attention_mask])
        # Gather perplexities from device
        ppl = [score.item() for score in ppl]

        return ppl

    def eval_generator(self, dialogues: List[Dict]) -> Tuple[float, int]:
        with torch.no_grad(), torch.autocast(self.device.type, enabled=self.mixed_precision):
            # Compute the PPL score for each utterance of each utterances
            ppl_scores = sum(
                [self.score_dialogue(dialogue['task_description'], dialogue['utterances']) for dialogue in dialogues],
                list()
            )
            # Get support (i.e., number of evaluated utterances)
            support = len(ppl_scores)
            # Average PPL scores
            ppl_score = np.mean(ppl_scores)

        return ppl_score, support

    def score_explanation(self, task_description: str, passages: List[Dict[str, str]]) -> List[float]:
        # Encode task description
        task_description_encodings = self.tokenizer(f'{task_description}', return_tensors='pt').to(self.device)
        # Prepare accumulators
        past_attention_mask = task_description_encodings.attention_mask
        past_key_values = self.model.transformer(**task_description_encodings).past_key_values
        # Perplexity accumulator
        ppl = []
        # Iterate over utterances
        for idx, passage in enumerate(passages):
            # Encode current passage-explanation pair
            input_ids, attention_mask = self.tokenizer(
                f'{passage["context_response"]}{passage["explanation"]}', return_tensors='pt'
            ).to(self.device).values()
            # Get current prompt length
            prompt_length = len(self.tokenizer(passage["context_response"]))
            # Get maximum response length
            max_response_len = self.tokenizer.model_max_length - attention_mask.size(0)
            # Target labels (they will be shifted automatically by NLL computation)
            labels = input_ids[:, prompt_length:max_response_len].clone()
            labels[~attention_mask[:, prompt_length:max_response_len].bool()] = IGNORE_INDEX
            # Get model predictions
            output = self.model(
                input_ids=input_ids,
                attention_mask=torch.hstack([past_attention_mask, attention_mask]),
                past_key_values=past_key_values
            )
            logits = output.logits[:, prompt_length:]
            # Compute sequence PPL
            ppl.append(self._nll(logits, labels, reduction='seqmean').squeeze().exp())
        # Gather perplexities from device
        ppl = [score.item() for score in ppl]

        return ppl

    def eval_explanations(self, dialogues: List[Dict]) -> Tuple[float, int]:
        with torch.no_grad(), torch.autocast(self.device.type, enabled=self.mixed_precision):
            # Compute the PPL score for each utterance of each utterances
            ppl_scores = sum(
                [self.score_explanation(dialogue['task_description'], dialogue['passages']) for dialogue in dialogues],
                list()
            )
            # Get support (i.e., number of evaluated utterances)
            support = len(ppl_scores)
            # Average PPL scores
            ppl_score = np.mean(ppl_scores)

        return ppl_score, support

    def predict_global_label(self, approach: Literal['posterior', 'infilling'], **kwargs) -> Tuple[int, int]:
        assert approach == 'infilling' or approach == 'infilling', f'Unsupported label type: \'{approach}\''
        # Accumulator for logits
        cls_logits = torch.empty(0, dtype=torch.float, device=self.device)
        # Encode common prefix
        prefix_encodings = self.tokenizer(
            kwargs['prompt'] if approach == 'posterior' else f'{kwargs["task_description"]}\n\n',
            return_tensors='pt',
            padding=True
        ).to(self.device)
        # Prepare accumulators
        past_attention_mask = prefix_encodings.attention_mask
        past_key_values = self.model.transformer(**prefix_encodings).past_key_values
        # Target label
        try:
            if approach == 'posterior':
                y_true = [kwargs['target'] in annotation for annotation in kwargs['annotations']].index(True)
            else:
                y_true = [kwargs['target'] in starter for starter in kwargs['starter']].index(True)
        except ValueError:
            y_true = -1
        # Get in memory elements
        in_mem = self.in_mem if self.in_mem is not None else len(kwargs['annotations' if approach == 'posterior' else 'starter'])
        # Iterate over batches of annotations
        for s_idx in range(0, len(kwargs['annotations' if approach == 'posterior' else 'starter']), in_mem):
            # Get last index
            e_idx = min(s_idx + in_mem, len(kwargs['annotations' if approach == 'posterior' else 'starter']))
            # Prepare current inputs
            if approach == 'posterior':
                input_ids, attention_mask = self.tokenizer(
                    kwargs['annotations'][s_idx:e_idx], return_tensors='pt', padding=True
                ).to(self.device).values()
            else:
                input_ids, attention_mask = self.tokenizer(
                    [f'{starter}\n\n{str().join(kwargs["utterances"])}' for starter in kwargs['starter'][s_idx:e_idx]],
                    return_tensors='pt',
                    padding=True
                ).to(self.device).values()
            # Target labels (they will be shifted automatically by NLL computation)
            labels = input_ids.clone()
            labels[~attention_mask.bool()] = IGNORE_INDEX
            # Prepare past
            past = tuple(
                (k.repeat(e_idx - s_idx, 1, 1, 1), v.repeat(e_idx - s_idx, 1, 1, 1))
                for (k, v) in past_key_values
            )
            past_attention = past_attention_mask.repeat(e_idx - s_idx, 1)
            # Get model predictions
            output = self.model(
                input_ids=input_ids,
                attention_mask=torch.hstack([past_attention, attention_mask]),
                past_key_values=past
            )
            # Get raw token scores
            gen_logits = output.logits
            # Compute sequence cumulative log-likelihood # NOTE the minus sign
            cls_logits = torch.cat([cls_logits, - self._nll(gen_logits, labels, reduction='seqsum').squeeze()])
        # Exp-normalisation
        # cls_logits -= cls_logits.max()
        # Predict class as argmax
        y_pred = torch.argmax(cls_logits).item()

        return y_true, y_pred

    def predict_local_label(self, approach: Literal['posterior', 'infilling', 'prediction'], **kwargs) -> List[Tuple[int, int]]:
        # Accumulator for true label
        y_true = []
        #
        if approach == 'posterior':
            # Accumulator for logits
            cls_logits = torch.empty((0, len(kwargs['passages'][0]['annotations'])), dtype=torch.float, device=self.device)
            # Encode common prefix
            prefix_encodings = self.tokenizer(kwargs['task_description'], return_tensors='pt').to(self.device)
            # Prepare accumulators
            past_attention_mask = prefix_encodings.attention_mask
            past_key_values = self.model.transformer(**prefix_encodings).past_key_values
            # Iterate over passages to label
            for passage in kwargs['passages']:
                # Current logits accumulator
                tmp_cls_logits = torch.empty(0)
                # Encode current passage
                passage_input_ids, passage_attention_mask = self.tokenizer(
                    passage['context_response'], return_tensors='pt', padding=True
                ).to(self.device).values()
                # Prepare accumulators
                tmp_attention_mask = torch.hstack([past_attention_mask, passage_attention_mask])
                tmp_past_key_values = self.model.transformer(
                    input_ids=passage_input_ids, attention_mask=tmp_attention_mask, past_key_values=past_key_values
                ).past_key_values
                # Target label
                try:
                    y_true.append([passage['target'] in annotation for annotation in passage['annotations']].index(True))
                except ValueError:
                    y_true.append(-1)
                # Get in memory elements
                in_mem = self.in_mem if self.in_mem is not None else len(passage['annotations'])
                # Iterate over batches of annotations
                for s_idx in range(0, len(passage['annotations']), in_mem):
                    # Get last index
                    e_idx = min(s_idx + in_mem, len(kwargs['annotations']))
                    # Prepare current inputs
                    input_ids, attention_mask = self.tokenizer(
                        passage['annotations'][s_idx:e_idx], return_tensors='pt', padding=True
                    ).to(self.device).values()
                    # Target labels (they will be shifted automatically by NLL computation)
                    labels = input_ids.clone()
                    labels[~attention_mask.bool()] = IGNORE_INDEX
                    # Prepare past
                    past = tuple(
                        (k.repeat(e_idx - s_idx, 1, 1, 1), v.repeat(e_idx - s_idx, 1, 1, 1))
                        for (k, v) in tmp_past_key_values
                    )
                    past_attention = tmp_attention_mask.repeat(e_idx - s_idx, 1)
                    # Get model predictions
                    output = self.model(
                        input_ids=input_ids,
                        attention_mask=torch.hstack([past_attention, attention_mask]),
                        past_key_values=past
                    )
                    # Get raw token scores
                    gen_logits = output.logits
                    # Compute sequence cumulative log-likelihood # NOTE the minus sign
                    tmp_cls_logits = torch.hstack([tmp_cls_logits, - self._nll(gen_logits, labels, reduction='seqsum')])
                # Update accumulated logits
                cls_logits = torch.vstack([cls_logits, tmp_cls_logits])
        elif approach == 'infilling' or approach == 'prediction':
            # Accumulator for logits
            cls_logits = torch.empty((0, len(kwargs['utterances'][0]['prompts'])), dtype=torch.float, device=self.device)
            # Encode common prefix
            prefix_encodings = self.tokenizer(kwargs['task_description'], return_tensors='pt').to(self.device)
            # Prepare accumulators
            past_attention_mask = tmp_attention_mask = prefix_encodings.attention_mask
            past_key_values = tmp_past_key_values = self.model.transformer(**prefix_encodings).past_key_values
            # Parameters
            prefix_len: int = prefix_encodings.input_ids(1)
            ellipsis_len: int = len(self.tokenizer('...\n\n').input_ids)
            utterances_len = []
            context_len = 0
            utterances = []
            # Iterate over utterances to label
            for idx, utterance in enumerate(kwargs['utterances']):
                # Current logits accumulator
                tmp_cls_logits = torch.empty(0)
                # Encode correct prompt and utterance to compose context
                utterances.append((utterance["prompts"][y_true[-1]], utterance["text"]))
                utterance_input_ids, utterance_attention_mask = self.tokenizer(
                    f'{utterance["prompts"][y_true[-1]]} {utterance["text"]}', return_tensors='pt', padding=True
                ).to(self.device).values()
                # Get current response length
                utterances_len.append(utterance_input_ids.size(0))
                # Check if maximum length was reached and reset past
                if tmp_attention_mask.size(0) + utterances_len[-1] > self.tokenizer.model_max_length:
                    #
                    context_idx = 0
                    while context_idx < idx and context_len + ellipsis_len + prefix_len + utterances_len[-1] > self.tokenizer.model_max_length:
                        context_len -= utterances_len[context_idx]
                        context_idx += 1
                    # Encode and process the new context
                    context_input_ids, context_attention_mask = self.tokenizer(
                        f'...\n\n{str().join(f"{p} {u}" for p, u in utterances[context_idx:idx])}',
                        return_tensors='pt'
                    ).to(self.device).values()
                    # Update past
                    tmp_attention_mask = torch.hstack([past_attention_mask, context_attention_mask])
                    tmp_past_key_values = self.model.transformer(
                        input_ids=context_input_ids,
                        attention_mask=torch.hstack([past_key_values, context_attention_mask]),
                        past_key_values=tmp_past_key_values
                    ).past_key_values
                # Get maximum response length
                max_response_len = self.tokenizer.model_max_length - tmp_attention_mask.size(0)
                # Update past key values
                cache = self.model.transformer(
                    input_ids=utterance_input_ids,
                    attention_mask=torch.hstack([tmp_attention_mask, utterance_attention_mask]),
                    past_key_values=tmp_past_key_values
                ).past_key_values
                # Target label
                try:
                    y_true.append(
                        [utterance['target'] in prompt for prompt in utterance['prompts']].index(True))
                except ValueError:
                    y_true.append(-1)
                # Get in memory elements
                in_mem = self.in_mem if self.in_mem is not None else len(utterance['prompts'])
                # Iterate over batches of annotations
                for s_idx in range(0, len(utterance['prompts']), in_mem):
                    # Get last index
                    e_idx = min(s_idx + in_mem, len(utterance['prompts']))
                    # Prepare current inputs
                    if approach == 'prediction':
                        input_ids, attention_mask = self.tokenizer(
                            utterance['prompts'][s_idx:e_idx], return_tensors='pt', padding=True
                        ).to(self.device).values()
                    else:
                        input_ids, attention_mask = self.tokenizer(
                            [f'{prompt} {utterance["text"]}' for prompt in utterance['prompts'][s_idx:e_idx]],
                            return_tensors='pt',
                            padding=True
                        ).to(self.device).values()
                    # Target labels (they will be shifted automatically by NLL computation)
                    labels = input_ids.clone()
                    labels[~attention_mask.bool()] = IGNORE_INDEX
                    # Prepare past
                    past = tuple(
                        (k.repeat(e_idx - s_idx, 1, 1, 1), v.repeat(e_idx - s_idx, 1, 1, 1))
                        for (k, v) in tmp_past_key_values
                    )
                    past_attention = tmp_attention_mask.repeat(e_idx - s_idx, 1)
                    # Get model predictions
                    output = self.model(
                        input_ids=input_ids,
                        attention_mask=torch.hstack([past_attention, attention_mask]),
                        past_key_values=past
                    )
                    # Get raw token scores
                    gen_logits = output.logits
                    # Compute sequence cumulative log-likelihood # NOTE the minus sign
                    tmp_cls_logits = torch.hstack([tmp_cls_logits, - self._nll(gen_logits, labels, reduction='seqsum')])
                # Update accumulated logits
                cls_logits = torch.vstack([cls_logits, tmp_cls_logits])
                # Update accumulators
                tmp_attention_mask = torch.hstack([tmp_attention_mask, utterance_attention_mask])
                tmp_past_key_values = cache
        else:
            raise ValueError(f'Unsupported label type: \'{approach}\'')
        # Exp-normalisation
        # cls_logits -= cls_logits.max(dim=1)
        # Predict class as argmax
        y_pred = torch.argmax(cls_logits, dim=1).item()

        return list(*zip(y_true, y_pred))

    def predict_label(
            self,
            label_type: Literal['global', 'local'],
            approach: Literal['posterior', 'infilling', 'prediction'],
            **kwargs
    ) -> List[Tuple[int, int]]:
        if label_type == 'global':
            return [self.predict_global_label(approach, **kwargs)]
        elif label_type == 'local':
            return self.predict_local_label(approach, **kwargs)
        else:
            raise ValueError(f'Unsupported label type: \'{label_type}\'')

    def eval_discriminator(
            self,
            samples: List[Dict],
            label_type: Literal['global', 'local'],
            approach: Literal['posterior', 'infilling', 'prediction'],
            **kwargs
    ) -> Dict:
        with torch.no_grad(), torch.autocast(self.device.type, enabled=self.mixed_precision):
            # Compute the target and predicted labels for each sample
            y_true, y_pred = list(zip(*[sum(
                (self.predict_label(label_type, approach, **sample, **kwargs) for sample in samples), list()
            )]))

        return classification_report(y_true, y_pred, output_dict=True)
