with open('inference.py', 'r') as f:
    content = f.read()

old = '''    # Load model and tokenizer
    config = SignLanguageByT5Config.from_pretrained(model_config)
    model = SignLanguageByT5ForConditionalGeneration.from_pretrained(
        model_checkpoint,
        # config=config,
        cache_dir=os.path.join(output_dir, "cache"),
    )
    tokenizer = ByT5Tokenizer.from_pretrained(tokenizer_checkpoint)
    
    # Move model to appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()'''

new = '''    # Free up GPU memory before loading the translation model
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Load model and tokenizer
    config = SignLanguageByT5Config.from_pretrained(model_config)
    model = SignLanguageByT5ForConditionalGeneration.from_pretrained(
        model_checkpoint,
        # config=config,
        cache_dir=os.path.join(output_dir, "cache"),
        torch_dtype=torch.float16,
    )
    tokenizer = ByT5Tokenizer.from_pretrained(tokenizer_checkpoint)
    
    # Move model to appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()'''

if old in content:
    content = content.replace(old, new, 1)
    with open('inference.py', 'w') as f:
        f.write(content)
    print("Patched successfully")
else:
    print("Pattern not found - need exact text match check")
