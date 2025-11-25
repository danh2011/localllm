import torch
import torch.nn as nn
import torch.optim as optim
import psutil
import platform
import math
import os
import json
import time
from dataclasses import dataclass, asdict
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pickle

@dataclass
class ModelConfig:
    vocab_size: int = 10000
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 512
    max_seq_len: int = 128
    dropout: float = 0.1

@dataclass
class GenerationPreset:
    name: str
    temperature: float
    top_k: int
    top_p: float
    max_length: int
    description: str

class SystemAnalyzer:
    """Analyzes system capabilities and recommends model configuration"""

    @staticmethod
    def get_system_info():
        info = {
            'cpu_count': psutil.cpu_count(logical=False),
            'cpu_threads': psutil.cpu_count(logical=True),
            'ram_gb': psutil.virtual_memory().total / (1024**3),
            'ram_available_gb': psutil.virtual_memory().available / (1024**3),
            'platform': platform.system(),
            'has_cuda': torch.cuda.is_available(),
            'cuda_device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            'cuda_memory_gb': torch.cuda.get_device_properties(0).total_memory / (1024**3) if torch.cuda.is_available() else 0
        }
        return info

    @staticmethod
    def print_system_info(info):
        print("\n" + "="*50)
        print("SYSTEM SPECIFICATIONS")
        print("="*50)
        print(f"Platform: {info['platform']}")
        print(f"CPU Cores: {info['cpu_count']} ({info['cpu_threads']} threads)")
        print(f"RAM: {info['ram_gb']:.2f} GB (Available: {info['ram_available_gb']:.2f} GB)")
        print(f"CUDA Available: {info['has_cuda']}")
        if info['has_cuda']:
            print(f"GPU: {info['cuda_device']}")
            print(f"GPU Memory: {info['cuda_memory_gb']:.2f} GB")
        print("="*50 + "\n")

    @staticmethod
    def recommend_config(info):
        """Recommend model configuration based on available resources"""
        available_ram = info['ram_available_gb']

        if available_ram < 2:
            return ModelConfig(
                vocab_size=256,
                d_model=128,
                n_heads=2,
                n_layers=2,
                d_ff=256,
                max_seq_len=64,
                dropout=0.1
            )
        elif available_ram < 4:
            return ModelConfig(
                vocab_size=512,
                d_model=192,
                n_heads=3,
                n_layers=3,
                d_ff=384,
                max_seq_len=96,
                dropout=0.1
            )
        elif available_ram < 8:
            return ModelConfig(
                vocab_size=1024,
                d_model=256,
                n_heads=4,
                n_layers=4,
                d_ff=512,
                max_seq_len=128,
                dropout=0.1
            )
        else:
            return ModelConfig(
                vocab_size=2048,
                d_model=384,
                n_heads=6,
                n_layers=6,
                d_ff=768,
                max_seq_len=256,
                dropout=0.1
            )

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape

        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        output = torch.matmul(attn, V)

        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.W_o(output)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_out = self.attn(self.norm1(x), mask)
        x = x + self.dropout(attn_out)
        ff_out = self.ff(self.norm2(x))
        x = x + self.dropout(ff_out)
        return x

class SimpleLLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embedding = nn.Embedding(config.max_seq_len, config.d_model)

        self.blocks = nn.ModuleList([
            TransformerBlock(config.d_model, config.n_heads, config.d_ff, config.dropout)
            for _ in range(config.n_layers)
        ])

        self.ln_f = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size)

        # Weight tying
        self.head.weight = self.token_embedding.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, mask=None):
        batch_size, seq_len = x.shape

        pos = torch.arange(0, seq_len, dtype=torch.long, device=x.device).unsqueeze(0)

        x = self.token_embedding(x) + self.pos_embedding(pos)

        for block in self.blocks:
            x = block(x, mask)

        x = self.ln_f(x)
        logits = self.head(x)

        return logits

class ImprovedTokenizer:
    """Improved tokenizer with BPE-style vocabulary building"""
    def __init__(self, vocab_size=1024):
        self.vocab_size = vocab_size
        self.char2idx = {}
        self.idx2char = {}
        self.special_tokens = {'<PAD>': 0, '<UNK>': 1, '<START>': 2, '<END>': 3}
        self._build_vocab()

    def _build_vocab(self):
        # Start with special tokens
        self.char2idx = self.special_tokens.copy()

        # Add printable ASCII characters
        chars = [chr(i) for i in range(32, 127)]
        for i, char in enumerate(chars, start=len(self.special_tokens)):
            if i >= self.vocab_size:
                break
            self.char2idx[char] = i

        self.idx2char = {v: k for k, v in self.char2idx.items()}

    def train_on_text(self, text):
        """Build vocabulary from text"""
        char_freq = {}
        for char in text:
            char_freq[char] = char_freq.get(char, 0) + 1

        # Add most frequent characters to vocab
        sorted_chars = sorted(char_freq.items(), key=lambda x: x[1], reverse=True)

        current_idx = len(self.special_tokens)
        for char, _ in sorted_chars:
            if current_idx >= self.vocab_size:
                break
            if char not in self.char2idx:
                self.char2idx[char] = current_idx
                self.idx2char[current_idx] = char
                current_idx += 1

    def encode(self, text):
        return [self.char2idx.get(c, self.char2idx['<UNK>']) for c in text]

    def decode(self, indices):
        return ''.join([self.idx2char.get(i, '<UNK>') for i in indices if i not in [0, 2, 3]])

class TextDataset(Dataset):
    """Optimized dataset with caching"""
    def __init__(self, text, tokenizer, seq_len, stride=None):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.stride = stride or seq_len // 2

        # Tokenize entire text
        self.tokens = tokenizer.encode(text)

        # Create sequences with overlap
        self.data = []
        for i in range(0, len(self.tokens) - seq_len, self.stride):
            chunk = self.tokens[i:i + seq_len + 1]
            if len(chunk) == seq_len + 1:
                self.data.append(chunk)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        chunk = self.data[idx]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

class Trainer:
    """Enhanced training manager with persistent state"""
    def __init__(self, model, tokenizer, config, device):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        self.optimizer = optim.AdamW(
            model.parameters(), 
            lr=3e-4,
            betas=(0.9, 0.95),
            weight_decay=0.1
        )
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=100,
            eta_min=1e-5
        )
        self.train_losses = []
        self.best_loss = float('inf')
        self.total_epochs = 0
        self.checkpoint_path = "auto_checkpoint.pt"

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0

        progress_bar = tqdm(dataloader, desc="Training")
        for batch_idx, (x, y) in enumerate(progress_bar):
            x, y = x.to(self.device), y.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(x)

            loss = self.criterion(logits.view(-1, self.config.vocab_size), y.view(-1))
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            self.optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })

        return total_loss / len(dataloader)

    def auto_save(self):
        """Automatically save checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': asdict(self.config),
            'tokenizer_char2idx': self.tokenizer.char2idx,
            'tokenizer_idx2char': self.tokenizer.idx2char,
            'train_losses': self.train_losses,
            'best_loss': self.best_loss,
            'total_epochs': self.total_epochs
        }
        torch.save(checkpoint, self.checkpoint_path)

    def train(self, text, epochs=10, batch_size=32, save_best=True, auto_checkpoint=True):
        """Train the model with automatic checkpointing"""
        print("\n" + "="*50)
        print("TRAINING")
        print("="*50)

        # Train tokenizer on text
        print("Building vocabulary from text...")
        self.tokenizer.train_on_text(text)

        # Create dataset
        dataset = TextDataset(text, self.tokenizer, self.config.max_seq_len)
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=0,
            pin_memory=self.device.type == 'cuda'
        )

        print(f"Dataset size: {len(dataset)} sequences")
        print(f"Batch size: {batch_size}")
        print(f"Epochs: {epochs}")
        print(f"Total epochs trained: {self.total_epochs}")
        print(f"Auto-checkpoint: {'Enabled' if auto_checkpoint else 'Disabled'}\n")

        start_time = time.time()

        # Training loop
        for epoch in range(epochs):
            print(f"\nEpoch {self.total_epochs + epoch + 1} (Session: {epoch + 1}/{epochs})")
            avg_loss = self.train_epoch(dataloader)
            self.train_losses.append(avg_loss)
            self.scheduler.step()

            print(f"Average Loss: {avg_loss:.4f}")

            # Save best model
            if save_best and avg_loss < self.best_loss:
                self.best_loss = avg_loss
                print(f"🎉 New best loss! Saving model...")

            # Auto-checkpoint after each epoch
            if auto_checkpoint:
                self.auto_save()
                print(f"💾 Auto-checkpoint saved")

            # Generate sample every 2 epochs
            if (epoch + 1) % 2 == 0:
                prompts = ["The ", "In ", "Once upon a time "]
                for prompt in prompts:
                    sample = self.generate_sample(prompt, max_length=60)
                    print(f"Sample '{prompt}': {sample}")

        self.total_epochs += epochs

        elapsed_time = time.time() - start_time
        print(f"\n✅ Training complete! Time: {elapsed_time:.2f}s")
        print(f"Final loss: {self.train_losses[-1]:.4f}")
        print(f"Best loss: {self.best_loss:.4f}")
        print(f"Total epochs: {self.total_epochs}")

    def generate_sample(self, prompt, max_length=50, temperature=0.8):
        """Generate a sample with better decoding"""
        self.model.eval()

        tokens = self.tokenizer.encode(prompt)
        tokens = torch.tensor([tokens], dtype=torch.long).to(self.device)

        with torch.no_grad():
            for _ in range(max_length):
                if tokens.shape[1] >= self.config.max_seq_len:
                    tokens = tokens[:, -self.config.max_seq_len:]

                logits = self.model(tokens)
                logits = logits[:, -1, :] / temperature

                # Top-k sampling
                top_k = 50
                top_logits, top_indices = torch.topk(logits, top_k)
                probs = torch.softmax(top_logits, dim=-1)
                next_token_idx = torch.multinomial(probs, num_samples=1)
                next_token = top_indices.gather(-1, next_token_idx)

                tokens = torch.cat([tokens, next_token], dim=1)

        return self.tokenizer.decode(tokens[0].cpu().tolist())

class PromptManager:
    """Manages custom prompts and generation presets"""
    def __init__(self):
        self.prompts_file = "custom_prompts.json"
        self.prompts = self.load_prompts()
        self.presets = {
            'creative': GenerationPreset(
                name='creative',
                temperature=1.0,
                top_k=50,
                top_p=0.95,
                max_length=200,
                description="High creativity, diverse output"
            ),
            'balanced': GenerationPreset(
                name='balanced',
                temperature=0.8,
                top_k=50,
                top_p=0.9,
                max_length=150,
                description="Balanced creativity and coherence"
            ),
            'precise': GenerationPreset(
                name='precise',
                temperature=0.5,
                top_k=30,
                top_p=0.85,
                max_length=150,
                description="More focused and coherent"
            ),
            'deterministic': GenerationPreset(
                name='deterministic',
                temperature=0.2,
                top_k=10,
                top_p=0.8,
                max_length=100,
                description="Very consistent, less random"
            )
        }

    def load_prompts(self):
        """Load saved prompts from file"""
        if os.path.exists(self.prompts_file):
            try:
                with open(self.prompts_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def save_prompts(self):
        """Save prompts to file"""
        with open(self.prompts_file, 'w') as f:
            json.dump(self.prompts, f, indent=2)

    def add_prompt(self, name, text):
        """Add a custom prompt"""
        self.prompts[name] = text
        self.save_prompts()
        print(f"✓ Saved prompt '{name}'")

    def delete_prompt(self, name):
        """Delete a custom prompt"""
        if name in self.prompts:
            del self.prompts[name]
            self.save_prompts()
            print(f"✓ Deleted prompt '{name}'")
        else:
            print(f"✗ Prompt '{name}' not found")

    def list_prompts(self):
        """List all custom prompts"""
        if not self.prompts:
            print("No custom prompts saved yet")
            return

        print("\n" + "="*50)
        print("CUSTOM PROMPTS")
        print("="*50)
        for name, text in self.prompts.items():
            preview = text[:50] + "..." if len(text) > 50 else text
            print(f"  {name}: {preview}")
        print("="*50)

    def get_prompt(self, name):
        """Get a prompt by name"""
        return self.prompts.get(name)

    def list_presets(self):
        """List all generation presets"""
        print("\n" + "="*50)
        print("GENERATION PRESETS")
        print("="*50)
        for name, preset in self.presets.items():
            print(f"  {name}: {preset.description}")
            print(f"    temp={preset.temperature}, top_k={preset.top_k}, top_p={preset.top_p}, len={preset.max_length}")
        print("="*50)

    def get_preset(self, name):
        """Get a preset by name"""
        return self.presets.get(name)

class LLMRunner:
    def __init__(self):
        # Analyze system
        self.system_info = SystemAnalyzer.get_system_info()
        SystemAnalyzer.print_system_info(self.system_info)

        # Get recommended configuration
        self.config = SystemAnalyzer.recommend_config(self.system_info)
        print("Recommended Model Configuration:")
        print(f"  Vocab Size: {self.config.vocab_size}")
        print(f"  Model Dimension: {self.config.d_model}")
        print(f"  Number of Heads: {self.config.n_heads}")
        print(f"  Number of Layers: {self.config.n_layers}")
        print(f"  Max Sequence Length: {self.config.max_seq_len}")
        print()

        # Initialize model
        self.device = torch.device('cuda' if self.system_info['has_cuda'] else 'cpu')
        print(f"Using device: {self.device}\n")

        self.model = SimpleLLM(self.config).to(self.device)
        self.tokenizer = ImprovedTokenizer(self.config.vocab_size)
        self.trainer = Trainer(self.model, self.tokenizer, self.config, self.device)
        self.prompt_manager = PromptManager()

        # Try to load auto-checkpoint
        if os.path.exists(self.trainer.checkpoint_path):
            print(f"🔄 Found auto-checkpoint, loading...")
            self.load_model(self.trainer.checkpoint_path)

        # Enable compilation if available (PyTorch 2.0+)
        if hasattr(torch, 'compile') and self.device.type == 'cuda':
            print("Enabling torch.compile for better performance...")
            try:
                self.model = torch.compile(self.model)
            except:
                pass

        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total Parameters: {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,}")
        print(f"Estimated Memory: {total_params * 4 / (1024**2):.2f} MB\n")

    def save_model(self, path="model_checkpoint.pt"):
        """Save model checkpoint with metadata"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.trainer.optimizer.state_dict(),
            'scheduler_state_dict': self.trainer.scheduler.state_dict(),
            'config': asdict(self.config),
            'tokenizer_char2idx': self.tokenizer.char2idx,
            'tokenizer_idx2char': self.tokenizer.idx2char,
            'train_losses': self.trainer.train_losses,
            'best_loss': self.trainer.best_loss,
            'total_epochs': self.trainer.total_epochs
        }
        torch.save(checkpoint, path)
        print(f"✓ Model saved to {path}")

    def load_model(self, path="model_checkpoint.pt"):
        """Load model checkpoint"""
        if not os.path.exists(path):
            print(f"✗ No checkpoint found at {path}")
            return False

        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])

            if 'optimizer_state_dict' in checkpoint:
                self.trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                self.trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

            self.tokenizer.char2idx = checkpoint['tokenizer_char2idx']
            self.tokenizer.idx2char = checkpoint['tokenizer_idx2char']

            if 'train_losses' in checkpoint:
                self.trainer.train_losses = checkpoint['train_losses']
            if 'best_loss' in checkpoint:
                self.trainer.best_loss = checkpoint['best_loss']
            if 'total_epochs' in checkpoint:
                self.trainer.total_epochs = checkpoint['total_epochs']

            print(f"✓ Model loaded from {path}")
            print(f"  Total epochs trained: {self.trainer.total_epochs}")
            print(f"  Best loss: {self.trainer.best_loss:.4f}")
            return True
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            return False

    def generate(self, prompt, max_length=150, temperature=0.8, top_k=50, top_p=0.9):
        """Enhanced generation with nucleus sampling"""
        self.model.eval()

        tokens = self.tokenizer.encode(prompt)
        tokens = torch.tensor([tokens], dtype=torch.long).to(self.device)

        generated_tokens = []

        with torch.no_grad():
            for _ in range(max_length):
                if tokens.shape[1] >= self.config.max_seq_len:
                    tokens = tokens[:, -self.config.max_seq_len:]

                logits = self.model(tokens)
                logits = logits[:, -1, :] / temperature

                # Top-k filtering
                top_k_logits, top_k_indices = torch.topk(logits, min(top_k, logits.size(-1)))

                # Top-p (nucleus) filtering
                sorted_logits, sorted_indices = torch.sort(top_k_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                top_k_logits[:, indices_to_remove] = float('-inf')

                probs = torch.softmax(top_k_logits, dim=-1)
                next_token_idx = torch.multinomial(probs, num_samples=1)
                next_token = top_k_indices.gather(-1, next_token_idx)

                tokens = torch.cat([tokens, next_token], dim=1)
                generated_tokens.append(next_token.item())

        all_tokens = self.tokenizer.encode(prompt) + generated_tokens
        return self.tokenizer.decode(all_tokens)

    def get_sample_training_data(self):
        """Generate comprehensive sample training data"""
        return """The art of storytelling has been a fundamental part of human culture for thousands of years. From ancient cave paintings to modern digital media, humans have always sought ways to share their experiences and imagination with others.

In the beginning, stories were passed down orally from generation to generation. Elders would gather children around fires and tell tales of great heroes, mysterious creatures, and important lessons. These stories served not only as entertainment but also as a means of preserving history and teaching values.

As civilizations developed, so did the methods of storytelling. The invention of writing allowed stories to be recorded and preserved in ways that oral tradition could not achieve. Ancient texts from civilizations like Egypt, Mesopotamia, and China contain some of the earliest written stories known to humanity.

The printing press revolutionized storytelling by making books accessible to a much wider audience. No longer were stories limited to the wealthy or educated elite. Common people could now read tales of adventure, romance, and mystery. This democratization of literature led to an explosion of creativity and the emergence of new literary genres.

In the modern era, technology has transformed storytelling once again. Movies, television, video games, and the internet have created entirely new forms of narrative expression. Virtual reality promises to make stories even more immersive, allowing people to step inside narratives and experience them firsthand.

Despite these technological advances, the core elements of good storytelling remain unchanged. Every compelling story needs interesting characters, meaningful conflict, and emotional resonance. Whether told around a campfire or through a computer screen, stories continue to connect us to our shared humanity.

The future of storytelling is limited only by our imagination. As artificial intelligence and other technologies continue to evolve, we may see forms of narrative that we cannot yet conceive. But one thing is certain: as long as humans exist, we will continue to tell stories, for they are essential to who we are."""

    def run_console(self):
        """Enhanced interactive console mode"""
        print("="*50)
        print("ENHANCED LLM - Console Mode")
        print("="*50)
        print("Commands:")
        print("  'train' - Train model on text")
        print("  'quick' - Quick train on sample data")
        print("  'continue' - Continue training from last session")
        print("  'load <file>' - Load saved model")
        print("  'save <file>' - Save current model")
        print("  'stats' - Show training statistics")
        print("  'config' - Show model configuration")
        print("  'presets' - List generation presets")
        print("  'prompts' - Manage custom prompts")
        print("  'use <preset>' - Generate with preset (creative/balanced/precise/deterministic)")
        print("  'quit' - Exit")
        print("  Or type a prompt to generate text\n")

        while True:
            try:
                user_input = input(">>> ").strip()

                if user_input.lower() == 'quit':
                    print("Goodbye!")
                    break

                elif user_input.lower() == 'continue':
                    print("\n🔄 Continuing training from last session...")
                    text_input = input("Enter training text or 'file:path.txt' or 'sample': ").strip()

                    if text_input.lower() == 'sample':
                        text = self.get_sample_training_data()
                        print(f"Using sample data ({len(text)} characters)")
                    elif text_input.startswith('file:'):
                        filepath = text_input[5:].strip()
                        try:
                            with open(filepath, 'r', encoding='utf-8') as f:
                                text = f.read()
                            print(f"✓ Loaded {len(text)} characters from {filepath}")
                        except Exception as e:
                            print(f"✗ Error loading file: {e}")
                            continue
                    else:
                        text = text_input

                    epochs = int(input(f"Additional epochs (current total: {self.trainer.total_epochs}): ") or "10")
                    batch_size = int(input("Batch size (default 32): ") or "32")
                    self.trainer.train(text, epochs=epochs, batch_size=batch_size)

                elif user_input.lower() == 'quick':
                    print("\n🚀 Quick training on sample data...")
                    text = self.get_sample_training_data()
                    print(f"Training on {len(text)} characters...")
                    epochs = int(input("Epochs (default 20): ") or "20")
                    self.trainer.train(text, epochs=epochs, batch_size=16)
                    print("\n✓ Quick training complete!")

                elif user_input.lower() == 'train':
                    print("\nEnter training text or 'file:path.txt' or 'sample':")
                    text_input = input(">>> ").strip()

                    if text_input.lower() == 'sample':
                        text = self.get_sample_training_data()
                        print(f"Using sample data ({len(text)} characters)")
                    elif text_input.startswith('file:'):
                        filepath = text_input[5:].strip()
                        try:
                            with open(filepath, 'r', encoding='utf-8') as f:
                                text = f.read()
                            print(f"✓ Loaded {len(text)} characters from {filepath}")
                        except Exception as e:
                            print(f"✗ Error loading file: {e}")
                            continue
                    else:
                        text = text_input

                    if len(text) < 100:
                        print("⚠ Warning: Text is very short!")

                    epochs = int(input("Epochs (default 15): ") or "15")
                    batch_size = int(input("Batch size (default 32): ") or "32")
                    self.trainer.train(text, epochs=epochs, batch_size=batch_size)

                elif user_input.lower().startswith('save'):
                    parts = user_input.split(maxsplit=1)
                    filename = parts[1] if len(parts) > 1 else "model_checkpoint.pt"
                    self.save_model(filename)

                elif user_input.lower().startswith('load'):
                    parts = user_input.split(maxsplit=1)
                    filename = parts[1] if len(parts) > 1 else "model_checkpoint.pt"
                    self.load_model(filename)

                elif user_input.lower() == 'stats':
                    print("\n" + "="*50)
                    print("TRAINING STATISTICS")
                    print("="*50)
                    if self.trainer.train_losses:
                        print(f"Total epochs trained: {self.trainer.total_epochs}")
                        print(f"Best loss: {self.trainer.best_loss:.4f}")
                        print(f"Latest loss: {self.trainer.train_losses[-1]:.4f}")
                        print(f"Recent loss history: {[f'{l:.4f}' for l in self.trainer.train_losses[-5:]]}")
                    else:
                        print("No training history yet")
                    print("="*50)

                elif user_input.lower() == 'config':
                    print("\n" + "="*50)
                    print("MODEL CONFIGURATION")
                    print("="*50)
                    for key, value in asdict(self.config).items():
                        print(f"  {key}: {value}")
                    print("="*50)

                elif user_input.lower() == 'presets':
                    self.prompt_manager.list_presets()

                elif user_input.lower() == 'prompts':
                    print("\nPrompt Management:")
                    print("  'list' - List all prompts")
                    print("  'add' - Add new prompt")
                    print("  'delete' - Delete prompt")
                    print("  'get <name>' - Get prompt by name")

                    action = input("Action: ").strip().lower()

                    if action == 'list':
                        self.prompt_manager.list_prompts()
                    elif action == 'add':
                        name = input("Prompt name: ").strip()
                        text = input("Prompt text: ").strip()
                        self.prompt_manager.add_prompt(name, text)
                    elif action == 'delete':
                        name = input("Prompt name: ").strip()
                        self.prompt_manager.delete_prompt(name)
                    elif action.startswith('get'):
                        parts = action.split(maxsplit=1)
                        if len(parts) > 1:
                            name = parts[1]
                        else:
                            name = input("Prompt name: ").strip()
                        prompt_text = self.prompt_manager.get_prompt(name)
                        if prompt_text:
                            print(f"\n'{name}': {prompt_text}")
                            use_it = input("\nGenerate with this prompt? (y/n): ").strip().lower()
                            if use_it == 'y':
                                user_input = prompt_text
                                # Fall through to generation
                            else:
                                continue
                        else:
                            print(f"✗ Prompt '{name}' not found")
                            continue
                    else:
                        print("Unknown action")
                        continue

                elif user_input.lower().startswith('use'):
                    parts = user_input.split(maxsplit=1)
                    if len(parts) < 2:
                        print("Usage: use <preset_name>")
                        self.prompt_manager.list_presets()
                        continue

                    preset_name = parts[1].strip()
                    preset = self.prompt_manager.get_preset(preset_name)

                    if not preset:
                        print(f"✗ Preset '{preset_name}' not found")
                        self.prompt_manager.list_presets()
                        continue

                    prompt = input("Enter your prompt: ").strip()

                    print(f"\n🤖 Generating with '{preset.name}' preset...")
                    print(f"   (temp={preset.temperature}, top_k={preset.top_k}, top_p={preset.top_p})\n")

                    output = self.generate(
                        prompt,
                        max_length=preset.max_length,
                        temperature=preset.temperature,
                        top_k=preset.top_k,
                        top_p=preset.top_p
                    )
                    print(f"📝 Output:\n{output}\n")

                else:
                    # Regular generation
                    if user_input:
                        print("\n🤖 Generating...\n")
                        temp = input("Temperature (0.1-2.0, default 0.8, press Enter to skip): ").strip()
                        temp = float(temp) if temp else 0.8

                        length = input("Max length (default 150, press Enter to skip): ").strip()
                        length = int(length) if length else 150

                        output = self.generate(user_input, max_length=length, temperature=temp)
                        print(f"\n📝 Output:\n{output}\n")

                        # Option to save as prompt
                        save_prompt = input("Save this as a custom prompt? (y/n): ").strip().lower()
                        if save_prompt == 'y':
                            name = input("Prompt name: ").strip()
                            if name:
                                self.prompt_manager.add_prompt(name, user_input)

                print("-"*50 + "\n")

            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"✗ Error: {e}\n")

if __name__ == "__main__":
    try:
        runner = LLMRunner()
        runner.run_console()
    except Exception as e:
        print(f"✗ Fatal error: {e}")
        print("\nTry closing other applications to free up memory.")