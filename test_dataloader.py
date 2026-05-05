# test_dataloader.py
from core.data_loader import DataLoader
from pathlib import Path

# Create sample data
sample_path = Path("data/raw/sample_complaints.csv")
sample_path.parent.mkdir(parents=True, exist_ok=True)

print("Creating sample data...")
DataLoader.create_sample_data(sample_path, num_samples=50)

print("\nLoading sample data...")
complaints, metadata = DataLoader.load_csv(sample_path)

print(f"✅ Loaded {len(complaints)} complaints")
print(f"\nFirst complaint: {complaints[0]}")
print(f"Metadata: {metadata[0] if metadata else 'None'}")

# Validate
print("\nValidating dataset...")
stats = DataLoader.validate_dataset(complaints)
for key, value in stats.items():
    print(f"  {key}: {value}")

print("\n✅ DataLoader test complete!")