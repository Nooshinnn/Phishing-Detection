import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from tqdm import tqdm
from transformers import AutoModelForMaskedLM

from model_and_data_utils import load_model, build_dataset, DEVICE

import sys

# Usage: python PDGAN_train_fixed.py [seed] [fluency_weight]
#   no args           -> seed 42, lambda 5.0 (the paper's configuration)
#   python PDGAN_train_fixed.py 100          -> seed 100, lambda 5.0
#   python PDGAN_train_fixed.py 42 4.0       -> seed 42, lambda 4.0
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 42
LAMBDA_ARG = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0
TAG = f"seed{SEED}" + ("" if LAMBDA_ARG == 5.0 else f"_lam{LAMBDA_ARG}")
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

print("=" * 100)
print("PDGAN GENERATOR TRAINING -- FIXED: trains on train_df, never touches test_df")
print("Using canonical checkpoint: original_best_model_fixed.pth")
print("=" * 100)

email_model, url_model, comm_model, email_tokenizer, url_tokenizer = load_model()
train_df, val_df, test_df = build_dataset()

# Freeze the victim model, but keep its embedding layer differentiable so gradients
# can flow through to the generator -- this is what makes PDGAN a genuine white-box
# attack (see forward_soft below). Everything else about the victim stays frozen.
for p in list(email_model.parameters()) + list(url_model.parameters()) + list(comm_model.parameters()):
    p.requires_grad = False
email_model.model.embeddings.word_embeddings.weight.requires_grad = True
print("Victim model frozen; embedding path kept differentiable for soft backprop.")


def forward_soft(model, soft_embeds, attention_mask=None):
    """Differentiable forward pass through the email encoder using soft
    (Gumbel-Softmax) token probabilities instead of discrete input_ids."""
    embedding_matrix = model.model.embeddings.word_embeddings.weight  # [vocab, 768]
    inputs_embeds = torch.matmul(soft_embeds, embedding_matrix)       # [B, T, 768]
    outputs = model.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
    pooled = outputs.last_hidden_state[:, 0, :]
    pooled = model.dropout(pooled)
    return model.fc(pooled)


class PDGAN_Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlm = AutoModelForMaskedLM.from_pretrained("distilbert-base-uncased").to(DEVICE)
    def forward(self, input_ids, attention_mask):
        return self.mlm(input_ids=input_ids, attention_mask=attention_mask).logits


generator = PDGAN_Generator()
g_optimizer = optim.Adam(generator.parameters(), lr=3e-5)

# FIX (mode collapse): the generator is initialized from a real MLM but had
# nothing keeping it linguistically plausible during adversarial fine-tuning --
# it collapsed onto spamming a small set of "magic words" that suppress
# P(phishing) regardless of context, producing incoherent text that achieves
# high evasion against the classifier but would never fool a human reader.
# Fix: keep a permanently FROZEN copy of the pretrained MLM as a fluency
# reference, and penalize the generator's output distribution for drifting
# away from what a plain, non-adversarial MLM would predict at each masked
# position. This keeps substitutions adversarial but still plausible.
reference_mlm = AutoModelForMaskedLM.from_pretrained("distilbert-base-uncased").to(DEVICE)
reference_mlm.eval()
for p in reference_mlm.parameters():
    p.requires_grad = False

FLUENCY_WEIGHT = LAMBDA_ARG  # (default 5.0; set from the command line)
# New-model data so far: 4.0 -> 21.44% evasion, incoherent.
                       # 6.0 -> 0.63% evasion, coherent. Trying the midpoint --
                       # same narrowing strategy as the original (old-model) sweep.
                       # Check epoch-5 samples before committing to the full run.

print(f"\nStarting PDGAN generator training on PHISHING-ONLY train_df examples...\n")

# FIX: the evasion threat model in this paper is "phishing classified as
# legitimate," not a symmetric attack on both classes. The original loss
# (-log(d_prob), unconditional on label) always pushed toward P(phishing)->1,
# which only ever attacks the LEGITIMATE class (forcing false positives -- an
# explicitly out-of-scope attack per the Limitations section) while doing
# nothing for, or actively reinforcing, correct predictions on phishing
# examples. Restrict to label==1 (phishing) rows so the generator is only
# ever trained against the attack this paper actually claims to evaluate.
train_df_phish = train_df[train_df['label'] == 1].reset_index(drop=True)
print(f"Restricting to phishing-labeled examples: {len(train_df_phish)} / {len(train_df)}")

N_EPOCHS = 20
BATCH_SIZE = 6

for epoch in range(N_EPOCHS):
    generator.train()
    total_g_loss = 0.0
    total_evasion_loss = 0.0
    total_fluency_loss = 0.0
    num_batches = 0

    for i in tqdm(range(0, len(train_df_phish), BATCH_SIZE), desc=f"Epoch {epoch + 1}/{N_EPOCHS}"):
        batch = train_df_phish.iloc[i:i + BATCH_SIZE]
        texts = batch['email_text'].tolist()

        inputs = email_tokenizer(
            texts, padding=True, truncation=True, max_length=256, return_tensors="pt"
        ).to(DEVICE)
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask

        mask_prob = 0.15  # reduced from 0.25 -- fewer simultaneous substitutions
        # per email gives the fluency term more relative influence per token
        # and leaves more of the original phishing text structurally intact.
        mask = (torch.rand_like(input_ids, dtype=torch.float32) < mask_prob)
        mask = mask & (input_ids != email_tokenizer.pad_token_id)

        input_ids_masked = input_ids.clone()
        input_ids_masked[mask] = email_tokenizer.mask_token_id

        gen_logits = generator(input_ids_masked, attention_mask)

        temperature = max(0.9 - epoch * 0.03, 0.3)
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(gen_logits) + 1e-9) + 1e-9)
        soft_probs = torch.softmax((gen_logits + gumbel_noise) / temperature, dim=-1)

        one_hot_real = torch.zeros_like(soft_probs).scatter_(-1, input_ids.unsqueeze(-1), 1.0)
        mask_expanded = mask.unsqueeze(-1).float()
        combined_embeds = mask_expanded * soft_probs + (1 - mask_expanded) * one_hot_real

        e_feat = forward_soft(email_model, combined_embeds, attention_mask)

        u_in = url_tokenizer(
            batch['url'].tolist(), padding=True, truncation=True, max_length=128, return_tensors="pt"
        ).to(DEVICE)
        with torch.no_grad():
            u_feat = url_model(u_in.input_ids, u_in.attention_mask)

        d_output = comm_model(e_feat, u_feat)
        d_prob = torch.sigmoid(d_output)  # P(phishing)

        # FIX: minimize P(phishing) for these (all label==1) examples -- i.e.
        # push toward evasion (predicted legitimate), the actual attack goal.
        # Was: g_loss = -mean(log(d_prob)), which pushed the OPPOSITE direction.
        evasion_loss = -torch.mean(torch.log(1 - d_prob + 1e-8))

        # FIX (mode collapse): fluency regularization -- KL(generator ||
        # frozen reference MLM), computed only at the masked positions, so
        # the generator can't drift arbitrarily far from plausible English
        # just to chase the evasion loss.
        with torch.no_grad():
            ref_logits = reference_mlm(input_ids=input_ids_masked, attention_mask=attention_mask).logits
        gen_log_probs = torch.log_softmax(gen_logits, dim=-1)
        ref_probs = torch.softmax(ref_logits, dim=-1)
        kl_per_token = torch.sum(ref_probs * (torch.log(ref_probs + 1e-9) - gen_log_probs), dim=-1)
        fluency_loss = (kl_per_token * mask.float()).sum() / mask.float().sum().clamp(min=1)

        g_loss = evasion_loss + FLUENCY_WEIGHT * fluency_loss

        g_optimizer.zero_grad()
        g_loss.backward()
        torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
        g_optimizer.step()

        total_g_loss += g_loss.item()
        total_evasion_loss += evasion_loss.item()
        total_fluency_loss += fluency_loss.item()
        num_batches += 1

    avg_loss = total_g_loss / max(num_batches, 1)
    avg_evasion = total_evasion_loss / max(num_batches, 1)
    avg_fluency = total_fluency_loss / max(num_batches, 1)
    print(f"Epoch {epoch + 1} | G Loss: {avg_loss:.4f} | "
          f"Evasion: {avg_evasion:.4f} | Fluency KL: {avg_fluency:.4f}")

    # FIX: checkpoint every 5 epochs so you can run PDGAN_eval_fixed.py against
    # an intermediate checkpoint and eyeball samples WITHOUT waiting for the
    # full 20 epochs -- much cheaper than discovering mode collapse at the end
    # again. Point the eval script at e.g. pdgan_generator_fixed_epoch5.pth.
    if (epoch + 1) % 5 == 0:
        ckpt_name = f"pdgan_generator_fixed_{TAG}_epoch{epoch + 1}.pth"
        torch.save(generator.state_dict(), ckpt_name)
        print(f"   [checkpoint saved: {ckpt_name}]")

torch.save(generator.state_dict(), f"pdgan_generator_fixed_{TAG}.pth")
print(f"\nSaved generator to pdgan_generator_fixed_{TAG}.pth")
if SEED == 42 and LAMBDA_ARG == 5.0:
    # the default name the other scripts (coordinated, hardening) load
    torch.save(generator.state_dict(), "pdgan_generator_fixed_seed42.pth")
    print("Also saved pdgan_generator_fixed_seed42.pth (default name)")
print("This generator has NEVER seen val_df or test_df -- run the evaluation")
print("script against PHISHING-LABELED test_df examples only to get a genuine")
print("evasion-attack-success number (see PDGAN_eval_fixed.py fix).")