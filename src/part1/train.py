from tqdm import tqdm
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import torch
from torchvision.ops import box_iou

def evaluate_detections(preds, targets):
    metric = MeanAveragePrecision(iou_type="bbox")
    metric.update(preds, targets)

    results = metric.compute()
    ious = []
    for i, pred in enumerate(preds):
      pred_boxes = pred['boxes'] 
      gt_boxes = targets[i]['boxes'] 

      if len(pred_boxes) == 0 or len(gt_boxes) == 0:
          continue 

      iou_matrix = box_iou(pred_boxes, gt_boxes) 
      max_ious_per_pred, _ = iou_matrix.max(dim=1) 
      ious.extend(max_ious_per_pred.tolist())
    mean_iou = sum(ious) / len(ious) if ious else 0
    print(f"Mean IoU on validation set: {mean_iou:.4f}")
    return {
        "mAP": results["map"].item(),
        "mAP_50": results["map_50"].item(),
        "mAP_75": results["map_75"].item(),
        "IoU": mean_iou
    }


def training(model, train_loader, valid_loader, num_epochs=10, device="cuda", save_dist="../../results/part1/best_model.pth"):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.1, patience=3, verbose=True
    )
    best_map = 0.0

    for epoch in range(num_epochs):
        model.train()
        train_losses = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for images, targets in pbar:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            optimizer.zero_grad()
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            losses.backward()
            optimizer.step()
            train_losses.append(losses.item())
            pbar.set_postfix({"loss": f"{losses.item():.4f}"})

        avg_loss = sum(train_losses) / len(train_losses)
        print(f"Epoch {epoch}, Train Loss: {avg_loss:.4f}")

        # Validation
        model.eval()
        val_detections = []
        val_targets = []
        with torch.no_grad():
            pbar2 = tqdm(valid_loader, desc=f"Validating Epoch {epoch+1}")
            for images, targets in pbar2:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                outputs = model(images)
                val_detections.extend(outputs)
                val_targets.extend(targets)

        metrics = evaluate_detections(val_detections, val_targets)
        print(
            f"Validation mAP: {metrics['mAP']:.4f}, mAP@50: {metrics['mAP_50']:.4f}, mAP@75: {metrics['mAP_75']:.4f}"
        )
        scheduler.step(metrics["mAP"])
        if metrics["mAP"] > best_map:
            best_map = metrics["mAP"]
            torch.save(model.state_dict(), save_dist)
            print("✅ Best model saved")
        torch.cuda.empty_cache()

def testing(test_model, test_loader, device="cuda"):
  test_model.eval()
  test_detections = []
  test_targets = []
  with torch.no_grad():
      pbar = tqdm(test_loader, desc=f"testing ...")
      for images, targets in pbar:
          images = [img.to(device) for img in images]
          outputs = test_model(images)
          test_targets.extend(targets)
          outputs[0]['scale'] = targets[0]['scale']
          _, outputs[0] = test_model.resize.convert(images[0],outputs[0])
          test_detections.extend(outputs)
  return test_detections, test_targets
