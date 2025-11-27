model = ElectraForSequenceClassification.from_pretrained(model_name, num_labels=5)
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
epochs = 4
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# 7. 학습 루프
epoch_results = []

print(f"\n### 학습 시작 (Total Epochs: {epochs}) ###")
for e in range(epochs):
    # Training
    model.train()
    total_train_loss = 0

    progress_bar = tqdm(train_dataloader, desc=f"Epoch {e + 1}/{epochs} (Train)", leave=False)
    for batch in progress_bar:
        b_input_ids, b_input_mask, b_labels = tuple(t.to(device) for t in batch)

        model.zero_grad()
        outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
        loss = outputs.loss
        total_train_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        progress_bar.set_postfix({'loss': loss.item()})

    avg_train_loss = total_train_loss / len(train_dataloader)

    # Validation
    model.eval()
    val_preds, val_true = [], []

    for batch in tqdm(val_dataloader, desc=f"Epoch {e + 1}/{epochs} (Val)", leave=False):
        b_input_ids, b_input_mask, b_labels = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            outputs = model(b_input_ids, attention_mask=b_input_mask)

        logits = outputs.logits
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        labels = b_labels.cpu().numpy()

        val_preds.extend(preds)
        val_true.extend(labels)

    # Calculate Metrics
    train_acc_dummy = 0  # 훈련 정확도는 시간 관계상 생략하거나 별도 계산 필요
    val_acc = np.mean(np.array(val_preds) == np.array(val_true))

    print(f"Epoch {e + 1}: Avg Loss={avg_train_loss:.4f}, Val Accuracy={val_acc:.4f}")
    epoch_results.append((avg_train_loss, 0, val_acc))  # Train Acc는 0으로 둠

# --- [C] 결과 시각화 ---

# 클래스 이름 정의 (5 Tier)
class_names = ['기능건의(0)', '기술오류(1)', '정책비판(2)', '공지(3)', '기타(4)']

print("\n===== 최종 모델 성능 평가 =====")
print(classification_report(val_true, val_preds, target_names=class_names, digits=4))

plot_confusion_matrix(val_true, val_preds, class_names)

# 모델 저장
save_path = "koelectra_soop_5tier_model.pt"
torch.save(model.state_dict(), save_path)
print(f"\n-> 모델 저장 완료: {save_path}")
