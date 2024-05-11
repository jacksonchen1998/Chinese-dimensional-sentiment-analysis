# %% [markdown]
# ## Load Library

# %%
import os
import pandas as pd
import torch
from sklearn.model_selection import train_test_split, KFold
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, BertForSequenceClassification, BertConfig, AdamW, get_linear_schedule_with_warmup
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

device = torch. device ("cuda:0" if torch. cuda. is_available else "cpu")

# %% [markdown]
# # Load dataset
# 
# - Training data from Chinese EmoBank CVAT
# - Testing data is private dataset

# %%
train_1 = "/root/home/Chinese-dimensional-sentiment-analysis/data/CVAT_1_SD.csv"
train_2 = "/root/home/Chinese-dimensional-sentiment-analysis/data/CVAT_2_SD.csv"
train_3 = "/root/home/Chinese-dimensional-sentiment-analysis/data/CVAT_3_SD.csv"
train_4 = "/root/home/Chinese-dimensional-sentiment-analysis/data/CVAT_4_SD.csv"
train_5 = "/root/home/Chinese-dimensional-sentiment-analysis/data/CVAT_5_SD.csv"
test = "/root/home/Chinese-dimensional-sentiment-analysis/data/test.csv"

# %% [markdown]
# # Create custom dataset
# 
# - Text
# - Mean
# - SD

# %%
# Create a custom dataset class
class RegressionDataset(Dataset):
    def __init__(self, texts, targets):
        self.texts = texts
        self.targets = torch.tensor(targets.values, dtype=torch.float)

        self.tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        inputs = self.tokenizer(
            text,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'targets': self.targets[idx]
        }

# %%
df1 = pd.read_csv(train_1, delimiter='\t')
df2 = pd.read_csv(train_2, delimiter='\t')
df3 = pd.read_csv(train_3, delimiter='\t')
df4 = pd.read_csv(train_4, delimiter='\t')
df5 = pd.read_csv(train_5, delimiter='\t')

dfs = [df1, df2, df3, df4, df5]

# %% [markdown]
# # Build Chinese Bert

# %%
# Define the custom model with BERT and a regression head
class BertRegressionModel(nn.Module):
    def __init__(self, bert_model_name, output_size):
        super(BertRegressionModel, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.regression_head = nn.Linear(self.bert.config.hidden_size, output_size)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # Get the [CLS] token output
        cls_output = outputs.last_hidden_state[:, 0, :]
        predictions = self.regression_head(cls_output)
        return predictions

# Create the model
model = BertRegressionModel("bert-base-chinese", output_size=2).to(device)

# %% [markdown]
# # K-fold (K=5)

# %%
# Define k-fold cross-validation function
def k_fold_cross_validation(model, dfs, n_splits=5, batch_size=2, num_epochs=3, learning_rate=2e-5):
    kf = KFold(n_splits=n_splits, shuffle=True)
    criterion = nn.MSELoss()

    all_losses = []
    for fold, (train_index, val_index) in enumerate(kf.split(dfs)):
        print(f"Fold {fold + 1}/{n_splits}")
        train_fold = dfs[fold]
        val_fold = dfs[fold]

        train_dataset = RegressionDataset(train_fold['Text'].tolist(), train_fold[['Valence_Mean', 'Valence_SD']])
        val_dataset = RegressionDataset(val_fold['Text'].tolist(), val_fold[['Valence_Mean', 'Valence_SD']])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        optimizer = AdamW(model.parameters(), lr=learning_rate)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_loader)*num_epochs)

        model.train()
        for epoch in range(num_epochs):
            running_loss = 0.0
            for batch in train_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                targets = batch['targets'].to(device)

                optimizer.zero_grad()
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                scheduler.step()

                running_loss += loss.item()

            epoch_loss = running_loss / len(train_loader)
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss}")

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                targets = batch['targets'].to(device)

                outputs = model(input_ids, attention_mask)
                val_loss = criterion(outputs, targets)
                val_losses.append(val_loss.item())

        fold_loss = sum(val_losses) / len(val_losses)
        print(f"Validation Loss: {fold_loss}")
        all_losses.append(fold_loss)

    avg_loss = sum(all_losses) / len(all_losses)
    print(f"Avg Validation Loss across all folds: {avg_loss}")

# %%
k_fold_cross_validation(model, dfs)

# %%
# Create a directory to save the model
model_save_path = "/root/home/Chinese-dimensional-sentiment-analysis/data/bert_regression_model_valence"
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)

# Save the model's state dictionary
torch.save(model.state_dict(), f"{model_save_path}/model_state.pth")

from transformers import BertTokenizer

# Create a tokenizer from a pre-trained BERT model
# If you're using traditional Chinese, use 'bert-base-chinese'
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

# Save the tokenizer
tokenizer.save_pretrained(model_save_path)

# %% [markdown]
# # Load weight and inference

# %%
# Recreate the model architecture
loaded_model = BertRegressionModel("bert-base-chinese", output_size=2).to(device)

loaded_model.load_state_dict(torch.load(f"{model_save_path}/model_state.pth"))

# %%
# Define prediction function
def predict(model, data_loader):
    model.eval()
    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['targets'].to(device)

            outputs = model(input_ids, attention_mask)
            all_predictions.extend(outputs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    return all_predictions, all_targets

# Perform predictions and calculate MAE and Pearson correlation coefficient for each fold
all_maes = []
all_pearsons = []
for i, df in enumerate([df1, df2, df3, df4, df5]):
    val_dataset = RegressionDataset(df['Text'].tolist(), df[['Valence_Mean', 'Valence_SD']])
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

    val_predictions, val_targets = predict(loaded_model, val_loader)

    val_predictions = [pred[0] for pred in val_predictions]  # Considering only 'Valence_Mean'
    val_targets = [target[0] for target in val_targets]  # Considering only 'Valence_Mean'

    mae = mean_absolute_error(val_targets, val_predictions)
    pearson_corr, _ = pearsonr(val_targets, val_predictions)

    print(f"Fold {i+1} MAE: {mae}")
    print(f"Fold {i+1} Pearson Correlation Coefficient: {pearson_corr}")

    all_maes.append(mae)
    all_pearsons.append(pearson_corr)

# Calculate average MAE and Pearson correlation coefficient
avg_mae = sum(all_maes) / len(all_maes)
avg_pearson = sum(all_pearsons) / len(all_pearsons)
print(f"Average MAE: {avg_mae}")
print(f"Average Pearson Correlation Coefficient: {avg_pearson}")


# %% [markdown]
# # Visualizing MAE and $r$ value
# 
# - valence
# - arousal

# %%
# Define the folds
folds = [i+1 for i in range(len(all_maes))]

# Plot MAE
plt.figure(figsize=(10, 5))
plt.plot(folds, all_maes, marker='o', linestyle='-', color='b', label='MAE')
plt.xlabel('Fold')
plt.ylabel('MAE')
plt.title('MAE for each fold')
plt.xticks(folds)
plt.grid(True)
plt.legend()
plt.savefig("./image/bert_v_mae.png")

# Plot Pearson correlation coefficient
plt.figure(figsize=(10, 5))
plt.plot(folds, all_pearsons, marker='o', linestyle='-', color='r', label='Pearson')
plt.xlabel('Fold')
plt.ylabel('Pearson Correlation Coefficient')
plt.title('Pearson Correlation Coefficient for each fold')
plt.xticks(folds)
plt.grid(True)
plt.legend()
plt.savefig("./image/bert_v_r.png")

plt.show()


