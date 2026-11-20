+   5. Evaluation: Compare the model against analyst labels, FinBERT baselines, and simple lexicon-based models; analyse errors by sector, rating, and language.

+   6. Deployment: Freeze the 4-bit base model and save only the adapters for production; integrate with a RAG layer that provides transcript context for interactive use.

This type of PEFT setup is typical in practice: the institution retains a single, carefully vetted base model and configures multiple narrow adapters per desk or task (credit research, equity research, risk, compliance), thus balancing reuse, governance, and specialization.

Implementation-wise, this case study reuses the QLoRA configuration pattern from the previous subsection: load the model in 4-bit NF4, call prepare_model_for_kbit_training, attach a LoRAConfig, wrap the model with get_peft_model, and pass the adapted model to Trainer. The only case-study-specific objects are the earnings-call datasets and their label mapping:

```
# Reuse the QLoRA scaffold above with task-specific datasets.
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(
    prepare_model_for_kbit_training(model),
    lora_config,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=earnings_train,
    eval_dataset=earnings_val,
)
```

The expected inputs are tokenized transcript segments with labels such as positive, negative, and neutral; the expected outputs are adapter weights, validation metrics, calibration diagnostics, and an error-analysis report by sector, quarter, and speaker type.

While the case study illustrates how PEFT methods are applied in practice, a broader question remains: when should one choose full fine-tuning over LoRA or QLoRA?