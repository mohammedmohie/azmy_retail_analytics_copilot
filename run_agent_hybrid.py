#!/usr/bin/env python3
"""
Azmy Retail Analytics Copilot
Main entrypoint
"""
import json
import logging
import os
import sys
from pathlib import Path
import click
from typing import List, Dict, Any

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from agent.graph_hybrid import HybridRetailAgent


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('agent.log')
        ]
    )


def load_questions(batch_file: str) -> List[Dict[str, Any]]:
    """Load questions from JSONL file"""
    questions = []
    try:
        with open(batch_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    questions.append(json.loads(line))
        return questions
    except Exception as e:
        print(f"Error loading questions from {batch_file}: {e}")
        sys.exit(1)


def save_outputs(outputs: List[Dict[str, Any]], output_file: str):
    """Save outputs to JSONL file"""
    try:
        with open(output_file, 'w') as f:
            for output in outputs:
                f.write(json.dumps(output) + '\n')
        print(f"Results saved to {output_file}")
    except Exception as e:
        print(f"Error saving outputs to {output_file}: {e}")
        sys.exit(1)


def download_database():
    """Download the Northwind database if it doesn't exist"""
    db_path = "data/northwind.sqlite"
    
    if os.path.exists(db_path):
        print(f"Database already exists at {db_path}")
        return db_path
    
    print("Downloading Northwind database...")
    
    try:
        import requests
        url = "https://raw.githubusercontent.com/jpwhite3/northwind-SQLite3/main/dist/northwind.db"
        
        os.makedirs("data", exist_ok=True)
        
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(db_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"Database downloaded to {db_path}")
        
        # Create lowercase compatibility views
        create_views(db_path)
        
        return db_path
        
    except Exception as e:
        print(f"Error downloading database: {e}")
        print("Please manually download the database from:")
        print("https://raw.githubusercontent.com/jpwhite3/northwind-SQLite3/main/dist/northwind.db")
        print(f"And save it as {db_path}")
        sys.exit(1)


def create_views(db_path: str):
    """Create lowercase compatibility views"""
    try:
        import sqlite3
        
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            views_sql = """
            CREATE VIEW IF NOT EXISTS orders AS SELECT * FROM Orders;
            CREATE VIEW IF NOT EXISTS order_items AS SELECT * FROM "Order Details";
            CREATE VIEW IF NOT EXISTS products AS SELECT * FROM Products;
            CREATE VIEW IF NOT EXISTS customers AS SELECT * FROM Customers;
            """
            
            cursor.executescript(views_sql)
            print("Created lowercase compatibility views")
            
    except Exception as e:
        print(f"Error creating views: {e}")


@click.command()
@click.option('--batch', required=True, help='Path to JSONL file with questions')
@click.option('--out', required=True, help='Path to output JSONL file')
@click.option('--model', default='phi3.5:3.8b-mini-instruct-q4_K_M', 
              help='Model name for Ollama')
@click.option('--db-path', default='data/northwind.sqlite',
              help='Path to SQLite database')
@click.option('--docs-dir', default='docs',
              help='Path to documents directory')
def main(batch: str, out: str, model: str, db_path: str, docs_dir: str):
    """
    Retail Analytics Copilot - Run batch evaluation
    
    Example:
        python run_agent_hybrid.py --batch sample_questions_hybrid_eval.jsonl --out outputs_hybrid.jsonl
    """
    setup_logging()
    logger = logging.getLogger(__name__)
    
    print("🚀 Starting Retail Analytics Copilot")
    
    # Ensure database exists
    if not os.path.exists(db_path):
        print(f"Database not found at {db_path}")
        db_path = download_database()
    
    # Validate paths
    if not os.path.exists(batch):
        print(f"Error: Batch file not found: {batch}")
        sys.exit(1)
    
    if not os.path.exists(docs_dir):
        print(f"Error: Docs directory not found: {docs_dir}")
        sys.exit(1)
    
    # Load questions
    print(f"📋 Loading questions from {batch}")
    questions = load_questions(batch)
    print(f"Found {len(questions)} questions")
    
    # Initialize agent
    print(f"🤖 Initializing agent with model: {model}")
    try:
        agent = HybridRetailAgent(
            db_path=db_path,
            docs_dir=docs_dir,
            model_name=model
        )
        print("✅ Agent initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize agent: {e}")
        print(f"❌ Failed to initialize agent: {e}")
        sys.exit(1)
    
    # Process questions
    outputs = []
    
    for i, question_data in enumerate(questions):
        question_id = question_data.get('id', f'question_{i}')
        question = question_data.get('question', '')
        format_hint = question_data.get('format_hint', 'str')
        
        print(f"\n🔍 Processing question {i+1}/{len(questions)}: {question_id}")
        print(f"Q: {question}")
        
        try:
            result = agent.run(
                question=question,
                format_hint=format_hint,
                question_id=question_id
            )
            
            outputs.append(result)
            
            print(f"✅ Answer: {result['final_answer']}")
            print(f"📊 Confidence: {result['confidence']}")
            print(f"📚 Citations: {len(result['citations'])} sources")
            
        except Exception as e:
            logger.error(f"Error processing question {question_id}: {e}")
            
            # Add error result
            outputs.append({
                "id": question_id,
                "final_answer": None,
                "sql": "",
                "confidence": 0.0,
                "explanation": f"Processing error: {str(e)}",
                "citations": []
            })
            
            print(f"❌ Error: {e}")
    
    # Save outputs
    print(f"\n💾 Saving results to {out}")
    save_outputs(outputs, out)
    
    # Summary
    successful = sum(1 for output in outputs if output['final_answer'] is not None)
    print(f"\n📈 Summary:")
    print(f"   Total questions: {len(questions)}")
    print(f"   Successful: {successful}")
    print(f"   Failed: {len(questions) - successful}")
    print(f"   Success rate: {successful/len(questions)*100:.1f}%")
    
    print("\n🎉 Done!")


if __name__ == '__main__':
    main()
