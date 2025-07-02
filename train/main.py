import tiktoken
import os
import sys
from .participle import create_dataloader_v1, GPTDatasetV1
import torch
from module.GPTModule import text_to_tokens_ids, tokens_ids_to_text, generate_text_simple


tokenizer = tiktoken.get_encoding("gpt2")

def calculate_loss_batch(input_batch, target_batch, model,device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0,1), target_batch.flatten())

    return loss

def calculate_loss_loader(data_loader, model, device, num_batches=None):
    """
    计算数据加载器中的损失(训练集和验证集)
    Args:
        data_loader: 数据加载器
        model: 模型
        device: 设备
        num_batches: 计算的批次数
    """
    total_loss = 0.0
    if len(data_loader) == 0:
        return float("nan")  
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calculate_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item() # 累加损失
        else:
            break
    
    return total_loss / num_batches

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calculate_loss_loader(train_loader, model, device, eval_iter)
        val_loss = calculate_loss_loader(val_loader, model, device, eval_iter)
    model.train()
    return train_loss, val_loss

def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_tokens_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        tokens_ids = generate_text_simple(model, encoded, 50, context_size)
    
    decoded_text = tokens_ids_to_text(tokens_ids, tokenizer)
    print(decoded_text.replace("\n", "  "))
    model.train()

def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter, start_context, tokenizer):
    train_losses, val_losses, track_tokens_seen = [], [], [] # 训练损失，验证损失，跟踪token数
    tokens_seen, global_step = 0, -1
    
    print(f"开始训练，总共 {num_epochs} 个epoch，每 {eval_freq} 步评估一次")
    print(f"优化器: {type(optimizer).__name__}")
    print(f"学习率: {optimizer.param_groups[0]['lr']}")

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        batch_count = 0
        
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calculate_loss_batch(input_batch, target_batch, model, device)
            
            # 检查损失是否为 NaN 或无穷大
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"警告: 检测到异常损失值 {loss.item()}，跳过此批次")
                continue
                
            loss.backward()
            
            # 添加梯度裁剪防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()

            # 正确累加epoch损失
            epoch_loss += loss.item()
            batch_count += 1
            
            tokens_seen += input_batch.numel()
            global_step += 1

            # 修复评估频率判断
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"=== 评估结果 ===")
                print(f"Epoch {epoch+1}/{num_epochs}, Step {global_step}")
                print(f"训练损失: {train_loss:.4f}, 验证损失: {val_loss:.4f}")
                print(f"已处理token数: {tokens_seen}")
                print("================")

        # 每个epoch结束时的统计
        avg_epoch_loss = epoch_loss / batch_count if batch_count > 0 else 0
        print(f"Epoch {epoch+1} 完成，平均损失: {avg_epoch_loss:.4f}")
        
        # 生成样本文本
        print(f"Epoch {epoch+1} 生成样本:")
        generate_and_print_sample(model, tokenizer, device, start_context)
    return train_losses, val_losses, track_tokens_seen