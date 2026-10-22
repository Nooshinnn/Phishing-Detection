import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from tqdm import tqdm
from transformers import AutoModelForMaskedLM

from model_and_data_utils import load_model, build_dataset, DEVICE

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

print("=" * 100)
print("PDGAN GENERATOR TRAINING -- FIXED: trains on train_df, never touches test_df")
print("Using canonical checkpoint: original_best_model_fixed.pth")
print("=" * 100)

email_model, url_model, comm_model, email_tokenizer, url_tokenizer = load_model()
train_df, val_df, test_df = build_dataset()

for p in list(email_model.parameters()) + list(url_model.parameters()) + list(comm_model.parameters()):
    p.requires_grad = False
email_model.model.embeddings.word_embeddings.weight.requires_grad = True
print("Victim model frozen; embedding path kept differentiable for soft backprop.")


def forward_soft(model, soft_embeds, attention_mask=None):
   
    embedding_matrix = model.model.embeddings.word_embeddings.weight  
    inputs_embeds = torch.matmul(soft_embeds, embedding_matrix)       
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

reference_mlm = AutoModelForMaskedLM.from_pretrained("distilbert-base-uncased").to(DEVICE)
reference_mlm.eval()
for p in reference_mlm.parameters():
    p.requires_grad = False

FLUENCY_WEIGHT = 5.0 

print(f"\nStarting PDGAN generator training on PHISHING-ONLY train_df examples...\n")

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

        mask_prob = 0.15 
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
        d_prob = torch.sigmoid(d_output) 

        evasion_loss = -torch.mean(torch.log(1 - d_prob + 1e-8))

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

    if (epoch + 1) % 5 == 0:
        ckpt_name = f"pdgan_generator_fixed_epoch{epoch + 1}.pth"
        torch.save(generator.state_dict(), ckpt_name)
        print(f"   [checkpoint saved: {ckpt_name}]")

torch.save(generator.state_dict(), "pdgan_generator_fixed.pth")
print("\nSaved generator to pdgan_generator_fixed.pth")
print("This generator has NEVER seen val_df or test_df -- run the evaluation")
print("script against PHISHING-LABELED test_df examples only to get a genuine")
print("evasion-attack-success number (see PDGAN_eval_fixed.py fix).")