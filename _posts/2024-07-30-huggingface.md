---
title: "1. Huggingface"
date: 2024-07-30 10:00:00 +0900
categories: ["Artificial Intelligence", "Library"]
tags: ["huggingface", "nlp", "llm"]
use_math: true
---

![alt text](/assets/img/post/library/huggingface.png)

# Huggingface

허깅페이스는~

## 1. Trainer
### 1) Preprocessing

```python
data = {"input_ids": [],
        "attention_mask": [],
        "labels": 1}
```

> - `input_ids`<br>
>   : 토큰화된 Input Text
> - `attention_mask`<br>
>   : 각 토큰이 실제 단어인지(1) 패딩 토큰(0)인지 나타냄
> - `labels`<br>
>   : 입력에 대한 Ground Truth를 저장, 멀티라벨일 경우 배열을 사용하여 표현
>   
> 허깅페이스의 주요 입력 형식은 위와 같고 이 형식을 지켜서 trainer에 입력해야 한다.<br>

### 2) Parameter

```python
trainer = Trainer(
    model = model,
    args = args,
    train_dataset = train_dataset['train'],
    eval_dataset = train_dataset['test'],
    compute_metrics = compute_metrics,
    data_collator = data_collator,
    tokenizer = tokenizer,
    formatting_func = formatting_func, # SFTTrainer에만 존재
    peft_config = lora_config,         # SFTTrainer에만 존재
)
```
※ SFTTrainer: Supervised Fine-Tuning Trainer

> #### ⅰ) model
>
> 허깅페이스 라이브러리에서 제공되는 Pretrained Model을 사용해도 되지만,<br>
> `torch.nn.Module`을 사용해도 된다.
> 
> ---
> #### ⅱ) args
>
> 학습률, 배치, 로깅옵션 등의 설정을 포함하는 파라미터들로 예시는 다음과 같다.
>
> ```python
> from transformers import TrainingArguments
> 
> training_args = TrainingArguments(
>     output_dir='./results',          # output directory
>     num_train_epochs=1,              # total number of training epochs
>     per_device_train_batch_size=1,   # batch size per device during training
>     per_device_eval_batch_size=10,   # batch size for evaluation
>     warmup_steps=1000,               # number of warmup steps for learning rate scheduler
>     weight_decay=0.01,               # strength of weight decay
>     logging_dir='./logs',            # directory for storing logs
>     logging_steps=200,               # How often to print logs
>     do_train=True,                   # Perform training
>     do_eval=True,                    # Perform evaluation
>     evaluation_strategy="epoch",     # evalute after eachh epoch
>     gradient_accumulation_steps=64,  # total number of steps before back propagation
>     fp16=True,                       # Use mixed precision
>     fp16_opt_level="02",             # mixed precision mode
>     run_name="ProBert-BFD-MS",       # experiment name
>     seed=3                           # Seed for experiment reproducibility 3x3
> )
> ```
>
> ---
> #### ⅲ) train_dataset
>
> ```python
> data = {"input_ids": [],
>         "attention_mask": [],
>         "labels": 1}
> ```
>
> 위의 형식을 지키는 데이터 셋들을 입력으로 넣어주어야 한다.<br>
> 즉, 다음과 같은 방식으로 모델에 전달할 수 있다.
>
> | dataset을 Tokenize한 후 전달 | dataset과 Tokenizer를 함께 전달 |
> | --- | --- |
> | dataset = dataset.map(lambda x: tokenizer(x)))<br> trainer = Trainer(<br> $\qquad$ model=model, <br> $\qquad$ args=args <br> $\qquad$ train_dataset = dataset<br> $\qquad$ data_collator=data_collator)| trainer = Trainer(<br> $\qquad$ model=model, <br> $\qquad$ args=args <br> $\qquad$ train_dataset = dataset<br> $\qquad$ data_collator=data_collator<br> $\qquad$ tokenizer = tokenizer) |
> 
>
> ---
> #### ⅳ) compute_metrics
>
> 평가 Metrics를 계산하는 함수로 pred를 input으로 받아 dictionary 형태로 return하는 함수
>
> ```python
> from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
> 
> def compute_metrics(pred):
>     labels = pred.label_ids
>     preds = pred.predictions.argmax(-1)
>     precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
>     acc = accuracy_score(labels, preds)
>     auc = roc_auc_score(labels, preds)
>     return {
>         'accuracy': acc,
>         'f1': f1,
>         'precision': precision,
>         'recall': recall,
>         'auroc': auc
>     }
> ```
> 
> ---
> #### ⅴ) [data_collator](https://huggingface.co/docs/transformers/v4.40.2/en/main_classes/data_collator#data-collator)
> 
> train_data의 원소들을 배치로 만들어주기 위해 사용되는 함수<br>
> ⅰ. tokenizer를 지정해주지 않은 경우: `default_data_collator()`가 사용<br>
> ⅱ. tokenizer를 지정해준 경우: `DataCollatorWithPadding`객체 사용
>
> tokenizer의 사용 여부에 따라 배치별로 마스킹이나 패딩등의 전처리를 수행해 주기도 함<br>
> 또한, template를 통해 text의 형식을 지정해 주는 역할을 하기도 함
> 
> - DefaultDataCollator<br>
>   : 특별한 전처리 없이 데이터를 배치로 묶기만 하는 data collator
> - DataCollatorWithPadding<br>
>   : 입력 데이터의 길이가 동일해지도록 패딩하여 배치로 만드는 data collator<br>
>   ($\rightarrow$ Text분류와 같은 작업에 사용됨)
> - DataCollatorForSeq2Seq<br>
>   : Source와 Target Text모두 패딩해주는 data collator<br>
>   ($\rightarrow$ 번역, 요약과 같은 Sequence to Sequence작업에 사용)
> - DataCollatorForCompletionOnlyLM<br>
>   : Masked Language Modelin과 같은 언어 모델링 task에 사용
>
> ※ Collator: 교정자
>
> ---
> #### ⅵ) tokenizer
>
> 입력 텍스트를 토큰화하는데 사용되는 토크나이저 객체
>
> - AutoTokenizer


### 3) Loss

```python
from torch import nn
from transformers import Trainer

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss (suppose one has 3 labels with different weights)
        loss_fct = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0, 3.0]))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss
```

> loss를 새로 만들어 최적화해야 하는 경우 Trainer클래스를 상속받아 새로운 Custom Trainer를 만들어 Compute_loss함수를 새로 작성해 주어야 한다.

---

## 2. [Tokenizer](https://wikidocs.net/166845)

![alt text](/assets/img/post/library/tokenizer.png)

NLP모델에서 텍스트 데이터를 토큰으로 변환해 주는 역할을 하는것이 Tokenizer이다

### 1) 종류

### 2) 동작과정

> | | |
> | --- | --- |
> | **1. 텍스트 분할** | **입력 텍스트를 토큰으로 분할한다.**<br> $\quad$`"Hugging Face is great!"`<br>$\quad \rightarrow$ `["Hugging", "Face", "is", "great", "!"]`<br>$\,$ |
> | **2. 토큰 변환** | **분할된 텍스트들을 고유한 숫자 ID로 변환한다.**<br>(각 토큰은 미리 정의된 Vocabulary를 통해 할당)<br>$\,$|
> | **3. 패딩 및 정규화** | **ⅰ. 문장의 길이를 고정된 길이로 맞춘다.**<br> $\quad$`["Hugging", "Face", "is", "great", "!"]`<br>$\quad \rightarrow$ `["Hugging", "Face", "is", "great", "!", "[PAD]", "[PAD]"]`<br><br> **ⅱ. 대소문자 변환 등의 정규화 작업을 수행한다.**<br> $\quad$`["Hugging", "Face", "is", "great", "!"]`<br>$\quad \rightarrow$ `["hugging", "face", "is", "great", "!"]`<br>$\,$ |
> | **4. 문장 부호 추가** | **모델이 문장의 시작과 끝을 인식할 수 있도록 시작 토큰과<br> 종료 토큰을 추가한다.**<br> (모델의 종류에 따라 사용하는 토큰이 달라진다.)<br><br>$\quad$ ⅰ. 대괄호(`[]`): BERT와 같은 모델에서 주로 사용<br>$\qquad$ ● `[CLS]`: 문장의 시작을 나타내는 토큰<br>$\qquad$ ● `[SEP]`: 문장의 끝을 나타내는 토큰<br>$\qquad$ ● `[INST]`: 대화형 모델에서 특정 지시를 수행하는 모델에서 사용 <br><br> $\quad$ ⅱ. 꺽쇠(`<>`): BART, GPT-2에서 주로 사용<br>$\qquad$ ● `<s>`: 문장의 시작을 나타내는 토큰<br>$\qquad$ ● `</s>`: 문장의 끝을 나타내는 토큰 <br>$\qquad$● `<<SYS>>`: 시스템의 역할이나 메시지를 구분하기 위해 사용 <br><br> 위의 두 종류의 토큰은 한종류만 사용할 필요는 없다.<br>$\,$ |
> | **5. Mask 생성** | **어텐션 마스크를 생성한다.** <br>$\quad$`["hugging", "face", "is", "great", "!"]`<br>$\quad \rightarrow$ `["Hugging", "Face", "is", "great", "!"]`<br> $\qquad$ `[1, 1, 1, 1, 1]`<br>$\,$ |
> 