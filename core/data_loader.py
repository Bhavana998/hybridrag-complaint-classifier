# core/data_loader.py - Complete Working Version
import pandas as pd
import json
import csv
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime
import logging
import random

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    """
    Data loader for complaint datasets in various formats
    Supports CSV, JSON, Excel, and text files
    """
    
    @staticmethod
    def load_csv(file_path: Path, text_column: str = 'complaint') -> Tuple[List[str], List[Dict]]:
        """
        Load complaints from CSV file
        
        Args:
            file_path: Path to CSV file
            text_column: Name of column containing complaint text
        
        Returns:
            Tuple of (complaints list, metadata list)
        """
        try:
            logger.info(f"Loading CSV from {file_path}")
            df = pd.read_csv(file_path)
            
            # Check if required column exists
            if text_column not in df.columns:
                available_cols = df.columns.tolist()
                raise ValueError(f"Column '{text_column}' not found. Available columns: {available_cols}")
            
            # Extract complaints
            complaints = df[text_column].fillna('').astype(str).tolist()
            
            # Extract metadata from all other columns
            metadata = []
            for _, row in df.iterrows():
                meta = {}
                for col in df.columns:
                    if col != text_column and pd.notna(row[col]):
                        meta[col] = str(row[col])
                metadata.append(meta)
            
            # Remove empty complaints
            valid_indices = [i for i, c in enumerate(complaints) if c and c.strip()]
            complaints = [complaints[i] for i in valid_indices]
            metadata = [metadata[i] for i in valid_indices]
            
            logger.info(f"✅ Loaded {len(complaints)} complaints from CSV")
            return complaints, metadata
            
        except Exception as e:
            logger.error(f"Error loading CSV: {str(e)}")
            raise
    
    @staticmethod
    def load_json(file_path: Path, text_key: str = 'complaint') -> Tuple[List[str], List[Dict]]:
        """
        Load complaints from JSON file
        
        Args:
            file_path: Path to JSON file
            text_key: Key name for complaint text in JSON objects
        
        Returns:
            Tuple of (complaints list, metadata list)
        """
        try:
            logger.info(f"Loading JSON from {file_path}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            complaints = []
            metadata = []
            
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        # Extract complaint text
                        complaint = item.get(text_key, item.get('text', item.get('complaint_text', '')))
                        if complaint and str(complaint).strip():
                            complaints.append(str(complaint))
                            # Store remaining fields as metadata
                            meta = {k: v for k, v in item.items() 
                                   if k not in [text_key, 'text', 'complaint_text']}
                            metadata.append(meta)
                    else:
                        if str(item).strip():
                            complaints.append(str(item))
                            metadata.append({})
            
            elif isinstance(data, dict):
                # Try to find array in dict
                for key in ['complaints', 'data', 'items', 'records']:
                    if key in data and isinstance(data[key], list):
                        for item in data[key]:
                            if isinstance(item, dict):
                                complaint = item.get(text_key, item.get('text', ''))
                                if complaint and str(complaint).strip():
                                    complaints.append(str(complaint))
                                    metadata.append({})
                        break
                else:
                    # Single complaint object
                    complaint = data.get(text_key, data.get('text', ''))
                    if complaint:
                        complaints.append(str(complaint))
                        metadata.append({k: v for k, v in data.items() if k != text_key})
            
            logger.info(f"✅ Loaded {len(complaints)} complaints from JSON")
            return complaints, metadata
            
        except Exception as e:
            logger.error(f"Error loading JSON: {str(e)}")
            raise
    
    @staticmethod
    def load_excel(file_path: Path, sheet_name: Optional[str] = None, 
                   text_column: str = 'complaint') -> Tuple[List[str], List[Dict]]:
        """
        Load complaints from Excel file
        
        Args:
            file_path: Path to Excel file
            sheet_name: Name of sheet (None for first sheet)
            text_column: Column name for complaint text
        
        Returns:
            Tuple of (complaints list, metadata list)
        """
        try:
            logger.info(f"Loading Excel from {file_path}")
            
            if sheet_name:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
            else:
                df = pd.read_excel(file_path)
            
            if text_column not in df.columns:
                available_cols = df.columns.tolist()
                raise ValueError(f"Column '{text_column}' not found. Available: {available_cols}")
            
            complaints = df[text_column].fillna('').astype(str).tolist()
            
            metadata = []
            for _, row in df.iterrows():
                meta = {}
                for col in df.columns:
                    if col != text_column and pd.notna(row[col]):
                        meta[col] = str(row[col])
                metadata.append(meta)
            
            # Remove empty complaints
            valid_indices = [i for i, c in enumerate(complaints) if c and c.strip()]
            complaints = [complaints[i] for i in valid_indices]
            metadata = [metadata[i] for i in valid_indices]
            
            logger.info(f"✅ Loaded {len(complaints)} complaints from Excel")
            return complaints, metadata
            
        except Exception as e:
            logger.error(f"Error loading Excel: {str(e)}")
            raise
    
    @staticmethod
    def load_text(file_path: Path, delimiter: str = '\n') -> Tuple[List[str], List[Dict]]:
        """
        Load complaints from plain text file (one per line)
        
        Args:
            file_path: Path to text file
            delimiter: Line delimiter (default: newline)
        
        Returns:
            Tuple of (complaints list, empty metadata list)
        """
        try:
            logger.info(f"Loading text from {file_path}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            complaints = [line.strip() for line in content.split(delimiter) 
                         if line.strip()]
            metadata = [{} for _ in complaints]
            
            logger.info(f"✅ Loaded {len(complaints)} complaints from text file")
            return complaints, metadata
            
        except Exception as e:
            logger.error(f"Error loading text file: {str(e)}")
            raise
    
    @staticmethod
    def load_auto(file_path: Path, **kwargs) -> Tuple[List[str], List[Dict]]:
        """
        Auto-detect file type and load complaints
        
        Args:
            file_path: Path to file
            **kwargs: Additional arguments for specific loaders
        
        Returns:
            Tuple of (complaints list, metadata list)
        """
        file_ext = file_path.suffix.lower()
        
        if file_ext == '.csv':
            return DataLoader.load_csv(file_path, **kwargs)
        elif file_ext == '.json':
            return DataLoader.load_json(file_path, **kwargs)
        elif file_ext in ['.xlsx', '.xls']:
            return DataLoader.load_excel(file_path, **kwargs)
        elif file_ext == '.txt':
            return DataLoader.load_text(file_path, **kwargs)
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")
    
    @staticmethod
    def save_results(results: List[Dict], output_path: Path, format: str = 'json'):
        """
        Save classification results to file
        
        Args:
            results: List of classification dictionaries
            output_path: Path to save file
            format: Output format ('json' or 'csv')
        """
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if format == 'json':
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, default=str)
                logger.info(f"✅ Results saved to {output_path} (JSON)")
            
            elif format == 'csv':
                df = pd.DataFrame(results)
                df.to_csv(output_path, index=False)
                logger.info(f"✅ Results saved to {output_path} (CSV)")
            
            else:
                raise ValueError(f"Unsupported format: {format}")
                
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
            raise
    
    @staticmethod
    def create_sample_data(output_path: Path, num_samples: int = 100, 
                          include_metadata: bool = True) -> None:
        """
        Create sample complaint data for testing
        
        Args:
            output_path: Path to save sample data
            num_samples: Number of sample complaints to generate
            include_metadata: Whether to include metadata columns
        """
        # Sample complaint templates
        complaint_templates = [
            "I was charged twice for my subscription of {amount}. Please refund the duplicate charge.",
            "The {product} {app} crashes every time I try to {action}. This is very frustrating.",
            "My package was supposed to arrive {days} days ago but tracking shows it's still stuck.",
            "Customer support was extremely rude and disconnected the call after {minutes} minutes.",
            "The {product} stopped working after only {days} days of use. Very poor quality.",
            "I cannot log into my account. The password reset email never arrives.",
            "Why was I charged a ${fee} late fee? I paid my bill on time.",
            "The website keeps timing out when I try to checkout my shopping cart.",
            "Driver delivered my package to the wrong address. I provided correct address.",
            "The product arrived damaged. The box was crushed and contents broken.",
            "I've been on hold for {minutes} minutes. No one is answering.",
            "The app update broke everything. Now nothing works.",
            "My account was hacked. Unauthorized purchases were made.",
            "The product quality is terrible. It fell apart after first use.",
            "I returned an item {weeks} weeks ago. Still no refund."
        ]
        
        # Categories with keywords
        categories = {
            "Billing": ["charged", "refund", "subscription", "fee", "payment", "bill"],
            "Technical": ["crashes", "app", "website", "error", "freeze", "bug"],
            "Shipping": ["package", "delivery", "tracking", "arrived", "shipping"],
            "Customer Service": ["support", "rude", "hold", "agent", "disconnected"],
            "Product Quality": ["stopped working", "quality", "damaged", "broke", "defective"],
            "Account": ["login", "password", "account", "reset", "hacked"]
        }
        
        data = []
        
        for i in range(num_samples):
            # Select random template
            template = random.choice(complaint_templates)
            
            # Generate random values
            amount = random.choice(["$49.99", "$99.99", "$29.99", "$19.99"])
            days = random.randint(1, 14)
            minutes = random.randint(15, 120)
            fee = random.choice([25, 50, 75, 100])
            weeks = random.randint(1, 4)
            product = random.choice(["headphones", "phone", "laptop", "tablet", "watch"])
            app = random.choice(["mobile app", "desktop app", "application"])
            action = random.choice(["upload a photo", "view orders", "make a payment", "load the page"])
            
            # Fill template
            complaint = template.format(
                amount=amount, days=days, minutes=minutes, fee=fee, 
                weeks=weeks, product=product, app=app, action=action
            )
            
            # Determine category based on keywords
            complaint_lower = complaint.lower()
            detected_category = "Other"
            for category, keywords in categories.items():
                if any(kw in complaint_lower for kw in keywords):
                    detected_category = category
                    break
            
            # Priority based on keywords
            priority = "Medium"
            if any(word in complaint_lower for word in ["urgent", "immediately", "emergency", "hacked"]):
                priority = "High"
            elif any(word in complaint_lower for word in ["minor", "small", "suggestion"]):
                priority = "Low"
            
            row = {
                "id": i + 1,
                "complaint": complaint,
                "category": detected_category,
                "priority": priority,
                "date": datetime.now().strftime("%Y-%m-%d"),
                "source": "sample_data"
            }
            
            if include_metadata:
                row["sentiment"] = random.choice(["negative", "neutral", "positive"])
                row["customer_id"] = f"CUST{random.randint(1000, 9999)}"
                row["region"] = random.choice(["US", "UK", "EU", "ASIA", "OTHER"])
            
            data.append(row)
        
        # Save to CSV
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        logger.info(f"✅ Created {num_samples} sample complaints at {output_path}")
        
        # Print statistics
        print(f"\n📊 Sample Data Statistics:")
        print(f"  Total complaints: {len(df)}")
        print(f"  Categories: {df['category'].value_counts().to_dict()}")
        print(f"  Priorities: {df['priority'].value_counts().to_dict()}")
    
    @staticmethod
    def merge_datasets(file_paths: List[Path], output_path: Path = None) -> Tuple[List[str], List[Dict]]:
        """
        Merge multiple datasets into one
        
        Args:
            file_paths: List of file paths to merge
            output_path: Optional path to save merged data
        
        Returns:
            Tuple of (merged complaints list, merged metadata list)
        """
        all_complaints = []
        all_metadata = []
        
        for file_path in file_paths:
            try:
                complaints, metadata = DataLoader.load_auto(file_path)
                all_complaints.extend(complaints)
                all_metadata.extend(metadata)
                logger.info(f"  Merged {len(complaints)} from {file_path.name}")
            except Exception as e:
                logger.error(f"  Failed to load {file_path.name}: {str(e)}")
        
        logger.info(f"✅ Merged total: {len(all_complaints)} complaints")
        
        if output_path:
            df = pd.DataFrame({
                'complaint': all_complaints,
                **{f'meta_{i}': [m.get(f'meta_{i}', '') for m in all_metadata] 
                   for i in range(5) if any(f'meta_{i}' in m for m in all_metadata)}
            })
            df.to_csv(output_path, index=False)
            logger.info(f"✅ Merged data saved to {output_path}")
        
        return all_complaints, all_metadata
    
    @staticmethod
    def validate_dataset(complaints: List[str]) -> Dict[str, Any]:
        """
        Validate dataset and return statistics
        
        Args:
            complaints: List of complaint texts
        
        Returns:
            Dictionary with validation statistics
        """
        total = len(complaints)
        empty = sum(1 for c in complaints if not c or not c.strip())
        duplicates = len(complaints) - len(set(complaints))
        avg_length = sum(len(c) for c in complaints) / total if total > 0 else 0
        short_entries = sum(1 for c in complaints if len(c) < 10)
        
        return {
            "total_complaints": total,
            "empty_entries": empty,
            "duplicates": duplicates,
            "avg_length": round(avg_length, 2),
            "short_entries": short_entries,
            "unique_complaints": len(set(complaints)),
            "is_valid": empty == 0 and duplicates < total * 0.1
        }


# Test function
if __name__ == "__main__":
    import tempfile
    
    print("="*50)
    print("Testing DataLoader")
    print("="*50)
    
    # Create sample data
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        sample_path = Path(f.name)
    
    print("\n1. Creating sample data...")
    DataLoader.create_sample_data(sample_path, num_samples=20)
    
    # Load the data
    print("\n2. Loading sample data...")
    complaints, metadata = DataLoader.load_csv(sample_path)
    print(f"   Loaded {len(complaints)} complaints")
    print(f"   First complaint: {complaints[0][:100]}...")
    
    # Validate
    print("\n3. Validating dataset...")
    stats = DataLoader.validate_dataset(complaints)
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # Save results example
    print("\n4. Testing save_results...")
    sample_results = [{"complaint": c[:50], "category": "Test"} for c in complaints[:3]]
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        results_path = Path(f.name)
    DataLoader.save_results(sample_results, results_path, format='json')
    print(f"   Results saved to {results_path}")
    
    # Clean up
    sample_path.unlink()
    results_path.unlink()
    
    print("\n✅ DataLoader test complete!")