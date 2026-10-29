"""
Fine-tune a YOLO model on your food dataset.

Usage:
    python train.py --data data.yaml --model yolo26n.pt --epochs 100
    python train.py --data data.yaml --model yolo11n.pt --epochs 100

Model size letters (n/s/m/l/x) trade speed for accuracy — start with 'n'
(nano) to iterate fast, move to 's' or 'm' once your pipeline works.
"""
import argparse
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data.yaml", help="Path to data.yaml")
    parser.add_argument(
        "--model",
        default="yolo26s.pt",
        help="Base checkpoint to fine-tune. Use yolo26n/s/m/l/x.pt for YOLO26, "
             "or yolo11n/s/m/l/x.pt for YOLO11. Downloaded automatically on first run. "
             "With ~50 classes, 's' (small) is a better starting point than 'n' (nano) — "
             "nano tends to underfit once you're past ~15-20 classes.",
    )
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None, help="'0' for GPU 0, 'cpu' to force CPU, blank = auto")
    parser.add_argument("--project", default="runs/train", help="Output dir for run artifacts")
    parser.add_argument("--name", default="food_detector", help="Run name")
    args = parser.parse_args()

    model = YOLO(args.model)

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        patience=20,       # early stop if val metric stalls
        plots=True,
    )

    # Quick validation summary after training
    metrics = model.val()
    print("\n=== Validation metrics ===")
    print(f"mAP50:    {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"\nBest weights saved to: {args.project}/{args.name}/weights/best.pt")


if __name__ == "__main__":
    main()
