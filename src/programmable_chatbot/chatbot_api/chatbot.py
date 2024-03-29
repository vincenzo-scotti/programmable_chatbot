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
            max_context_utterances: Optional[int] = None,
            device: Optional[torch.device] = None,
            mixed_precision: bool = True,
            in_mem: Optional[int] = None
    ):
        self.device: torch.device = device if device is not None else (
            torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        )
        self.model = model if isinstance(model, GPT2LMHeadModel) else GPT2LMHeadModel.from_pretrained(model)
        self.model = self.model.eval().to(self.device)
        self.tokenizer = GPT2Tokenizer.from_pretrained(
            tokenizer, pad_token='<|endoftext|>'
        ) if isinstance(tokenizer, str) else tokenizer
        self.max_response_length: int = max_response_length if max_response_length is not None else 256
        self.max_context_utterances: int = max_context_utterances if max_context_utterances is not None else 3
        self.mixed_precision: bool = mixed_precision
        self.in_mem: Optional[int] = in_mem

    def __call__(self, *args, **kwargs):
        ...

    def generate(
            self,
            context: List[str],  # NOTE context tokens lines are expected to end with '\n'
            prompt: str = '',
            task_description: Optional[str] = None,
            global_labels: Optional[str] = None,
            **kwargs
    ) -> str:  # TODO extend to handle better local labels
        dialogue = ''
        if task_description is not None:
            dialogue += task_description + '\n\n'
        if global_labels is not None:
            dialogue += global_labels + '\n\n'

        if len(dialogue) > 0:
            dialogue_len_tok = len(self.tokenizer(dialogue).input_ids)
        else:
            dialogue_len_tok = 0

        ellipsis = '...\n\n'
        ellipsis_len_tok = len(self.tokenizer(ellipsis).input_ids)

        context_utterances_len_tok = [len(self.tokenizer(utterance).input_ids) for utterance in context]
        context_len_tok = sum(context_utterances_len_tok)

        max_response_length = self.max_response_length - (len(self.tokenizer(prompt).input_ids) if len(prompt) > 0 else 0)
        max_context_len = self.tokenizer.model_max_length - max_response_length - dialogue_len_tok

        if context_len_tok > max_context_len:
            dialogue += ellipsis
            max_context_len = self.tokenizer.model_max_length - max_response_length - dialogue_len_tok - ellipsis_len_tok
            idx = 0
            while context_len_tok > max_context_len:
                context_len_tok -= context_utterances_len_tok[idx]
                idx += 1
            context = context[idx:]
        dialogue += ''.join(context + [prompt])

        with torch.no_grad(), torch.autocast(self.device.type, enabled=self.mixed_precision):
            input_ids = self.tokenizer(dialogue, return_tensors='pt').input_ids.to(self.device)
            response = self.tokenizer.decode(
                self.model.generate(input_ids, **kwargs)[0, input_ids.size(1):], skip_special_tokens=True
            )

        response, *_ = response.split('\n', 1)
        response = response.strip()

        return response

    def label_utterance(  # NOTE only one label at the time and only with targets is supported for now
            self,
            context: List[str],
            response: Union[str, Dict[str, Tuple[List[str], str]]],
            label: str,
            label_values: List[str],
            approach: Literal['posterior', 'infilling', 'prediction'],
            task_description: Optional[str] = None,
            output_proba: bool = False
    ):  # TODO manage possible length issue
        if approach == 'posterior':
            # Encode common prefix
            prefix_encodings = self.tokenizer(
                f'{task_description}\n\n'
                f'Context:\n\n{"".join(context[-self.max_context_utterances:])}\n\n'
                f'Response:\n\n{response}\n\n' if len(context) > 0 else
                f'{task_description}\n\nResponse:\n\n{response}\n\n'
                , return_tensors='pt'
            ).to(self.device)
            # Prepare accumulators
            past_attention_mask = prefix_encodings.attention_mask
            past_key_values = self.model.transformer(**prefix_encodings).past_key_values
            # Logits accumulator
            cls_logits = torch.empty(0, dtype=torch.float, device=self.device)
            # Get maximum completion length
            max_completion_len = self.tokenizer.model_max_length - past_attention_mask.size(1)
            # Get in memory elements
            in_mem = self.in_mem if self.in_mem is not None else len(label_values)
            # Iterate over batches of annotations
            for s_idx in range(0, len(label_values), in_mem):
                # Get last index
                e_idx = min(s_idx + in_mem, len(label_values))
                # Prepare current inputs
                input_ids, attention_mask = self.tokenizer(
                    [f'{label}: {label_values[idx]}.{self.tokenizer.eos_token}' for idx in range(s_idx, e_idx)],
                    return_tensors='pt', padding=True
                ).to(self.device).values()
                # Target labels (they will be shifted automatically by NLL computation)
                labels = input_ids.clone()
                labels[~attention_mask.bool()] = IGNORE_INDEX
                labels = labels[:, :max_completion_len]
                # Prepare past
                past = tuple(
                    (k.repeat(e_idx - s_idx, 1, 1, 1), v.repeat(e_idx - s_idx, 1, 1, 1))
                    for (k, v) in past_key_values
                )
                past_attention = past_attention_mask.repeat(e_idx - s_idx, 1)
                # Get model predictions
                output = self.model(
                    input_ids=input_ids[:, :max_completion_len],
                    attention_mask=torch.hstack([past_attention, attention_mask[:, :max_completion_len]]),
                    past_key_values=past
                )
                # Get raw token scores
                gen_logits = output.logits
                # Compute sequence cumulative log-likelihood # NOTE the minus sign
                cls_logits = torch.hstack([cls_logits, - self._nll(gen_logits, labels, reduction='seqsum')])
        elif approach == 'infilling' or approach == 'prediction':
            raise NotImplementedError()
        else:
            raise ValueError(f'Unsupported label type: \'{approach}\'')
        # Convert to string
        y_pred = label_values[torch.argmax(cls_logits).squeeze().item()]

        if output_proba:
            y_proba = {
                value: proba
                for value, proba in zip(
                    label_values,
                    torch.softmax(cls_logits.squeeze(), dim=0).view(-1).tolist()
                )
            }
            return y_pred, y_proba
        else:
            return y_pred

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
        context_idx = 0
        # Perplexity accumulator
        ppl = []
        # Iterate over utterances
        for idx, (prompt, utterance) in enumerate(utterances):
            # Encode current prompt-utterance pair
            utterance_input_ids, utterance_attention_mask = self.tokenizer(f'{prompt} {utterance}', return_tensors='pt').to(self.device).values()
            # Get current response length
            utterances_len.append(utterance_input_ids.size(1))
            # Get current prompt length
            prompt_length = len(self.tokenizer(prompt).input_ids)
            # Check if maximum length was reached and reset past
            if attention_mask.size(1) + utterances_len[-1] > self.tokenizer.model_max_length:
                #
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
            max_response_len = self.tokenizer.model_max_length - attention_mask.size(1)
            # Target labels (they will be shifted automatically by NLL computation)
            labels = utterance_input_ids[:, prompt_length:max_response_len].clone()
            labels[~utterance_attention_mask[:, prompt_length:max_response_len].bool()] = IGNORE_INDEX
            # Get model predictions
            output = self.model(
                input_ids=utterance_input_ids[:, :max_response_len],
                attention_mask=torch.hstack([attention_mask, utterance_attention_mask[:, :max_response_len]]),
                past_key_values=past_key_values
            )
            logits = output.logits[:, prompt_length:]
            # Compute sequence PPL
            ppl.append(self._nll(logits, labels, reduction='seqmean').squeeze().exp())
            # Update cache
            past_key_values = output.past_key_values
            # Update attention mask
            attention_mask = torch.hstack([attention_mask, utterance_attention_mask])
            # Update context length
            context_len += utterances_len[-1]
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
            prompt_length = len(self.tokenizer(passage["context_response"]).input_ids)
            # Get maximum response length
            max_completion_len = self.tokenizer.model_max_length - past_attention_mask.size(1)
            # Target labels (they will be shifted automatically by NLL computation)
            labels = input_ids.clone()
            labels[~attention_mask.bool()] = IGNORE_INDEX
            labels = labels[:, prompt_length:max_completion_len]
            # Get model predictions
            output = self.model(
                input_ids=input_ids[:, prompt_length:max_completion_len],
                attention_mask=torch.hstack([past_attention_mask, attention_mask[:, prompt_length:max_completion_len]]),
                past_key_values=past_key_values
            )
            logits = output.logits
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
        assert approach == 'infilling' or approach == 'posterior', f'Unsupported label type: \'{approach}\''
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
        except:
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
                tmp_cls_logits = torch.empty(0, dtype=torch.float, device=self.device)
                # Encode current passage
                passage_input_ids, passage_attention_mask = self.tokenizer(
                    passage['context_response'], return_tensors='pt', padding=True
                ).to(self.device).values()
                # Prepare accumulators
                tmp_attention_mask = torch.hstack([past_attention_mask, passage_attention_mask])
                tmp_past_key_values = self.model.transformer(
                    input_ids=passage_input_ids, attention_mask=tmp_attention_mask, past_key_values=past_key_values
                ).past_key_values
                # Get maximum copmletion length
                max_completion_len = self.tokenizer.model_max_length - tmp_attention_mask.size(1)
                # Target label
                try:
                    y_true.append([passage['target'] in annotation for annotation in passage['annotations']].index(True))
                except:
                    y_true.append(-1)
                # Get in memory elements
                in_mem = self.in_mem if self.in_mem is not None else len(passage['annotations'])
                # Iterate over batches of annotations
                for s_idx in range(0, len(passage['annotations']), in_mem):
                    # Get last index
                    e_idx = min(s_idx + in_mem, len(passage['annotations']))
                    # Prepare current inputs
                    input_ids, attention_mask = self.tokenizer(
                        passage['annotations'][s_idx:e_idx], return_tensors='pt', padding=True
                    ).to(self.device).values()
                    # Target labels (they will be shifted automatically by NLL computation)
                    labels = input_ids.clone()
                    labels[~attention_mask.bool()] = IGNORE_INDEX
                    labels = labels[:, :max_completion_len]
                    # Prepare past
                    past = tuple(
                        (k.repeat(e_idx - s_idx, 1, 1, 1), v.repeat(e_idx - s_idx, 1, 1, 1))
                        for (k, v) in tmp_past_key_values
                    )
                    past_attention = tmp_attention_mask.repeat(e_idx - s_idx, 1)
                    # Get model predictions
                    output = self.model(
                        input_ids=input_ids[:, :max_completion_len],
                        attention_mask=torch.hstack([past_attention, attention_mask[:, :max_completion_len]]),
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
            prefix_len: int = prefix_encodings.input_ids.size(1)
            ellipsis_len: int = len(self.tokenizer('...\n\n').input_ids)
            utterances_len = []
            context_len = 0
            utterances = []
            context_idx = 0
            # Iterate over utterances to label
            for idx, utterance in enumerate(kwargs['utterances']):
                # Target label
                try:
                    y_true.append([utterance['target'] in prompt for prompt in utterance['prompts']].index(True))
                except:
                    y_true.append(-1)
                # Current logits accumulator
                tmp_cls_logits = torch.empty(0, dtype=torch.float, device=self.device)
                # Encode correct prompt and utterance to compose context
                utterances.append((utterance["prompts"][y_true[-1]], utterance["text"]))
                utterance_input_ids, utterance_attention_mask = self.tokenizer(
                    f'{utterance["prompts"][y_true[-1]]} {utterance["text"]}', return_tensors='pt', padding=True
                ).to(self.device).values()
                # Get current response length
                utterances_len.append(utterance_input_ids.size(1))
                # Check if maximum length was reached and reset past
                if tmp_attention_mask.size(1) + utterances_len[-1] > self.tokenizer.model_max_length:
                    #
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
                        attention_mask=tmp_attention_mask,
                        past_key_values=past_key_values
                    ).past_key_values
                # Get maximum response length
                max_response_len = self.tokenizer.model_max_length - tmp_attention_mask.size(1)
                # Update past key values
                cache = self.model.transformer(
                    input_ids=utterance_input_ids[:, :max_response_len],
                    attention_mask=torch.hstack([tmp_attention_mask, utterance_attention_mask[:, :max_response_len]]),
                    past_key_values=tmp_past_key_values
                ).past_key_values
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
                    labels = labels[:, :max_response_len]
                    # Prepare past
                    past = tuple(
                        (k.repeat(e_idx - s_idx, 1, 1, 1), v.repeat(e_idx - s_idx, 1, 1, 1))
                        for (k, v) in tmp_past_key_values
                    )
                    past_attention = tmp_attention_mask.repeat(e_idx - s_idx, 1)
                    # Get model predictions
                    output = self.model(
                        input_ids=input_ids[:, :max_response_len],
                        attention_mask=torch.hstack([past_attention, attention_mask[:, :max_response_len]]),
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
                context_len += utterances_len[-1]
        else:
            raise ValueError(f'Unsupported label type: \'{approach}\'')
        # Exp-normalisation
        # cls_logits -= cls_logits.max(dim=1)
        # Predict class as argmax
        y_pred = torch.argmax(cls_logits, dim=1).view(-1).tolist()

        return list(zip(y_true, y_pred))

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
            y_true, y_pred = list(zip(*sum(
                (self.predict_label(label_type, approach, **sample, **kwargs) for sample in samples), list()
            )))

            y_true = np.array(y_true)
            y_pred = np.array(y_pred)
            # Remove missing labels
            mask = y_true != -1
            y_true = y_true[mask]
            y_pred = y_pred[mask]

        return classification_report(y_true, y_pred, output_dict=True)
