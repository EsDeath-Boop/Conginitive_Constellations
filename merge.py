import json
import uuid
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from datetime import datetime
import os
import re
from collections import Counter

# Configuration
SIM_THRESHOLD = 0.3  # Lower threshold for TF-IDF (adjust as needed)
OUTPUT_FILE = "chat_map.json"
MIN_TEXT_LENGTH = 20
MAX_TEXT_LENGTH = 2000

def load_messages(filepath, account_label=""):
    """Extract messages from a ChatGPT export file."""
    print(f"Loading messages from {filepath}...")
    
    if not os.path.exists(filepath):
        print(f"Warning: File {filepath} not found. Skipping.")
        return []
    
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON from {filepath}: {e}")
        return []

    messages = []
    for conv_idx, conv in enumerate(data):
        title = conv.get("title", f"Untitled Conversation {conv_idx}")
        mapping = conv.get("mapping", {})
        conv_id = conv.get("id", str(uuid.uuid4()))
        create_time = conv.get("create_time", datetime.now().timestamp())
        
        for node_id, node in mapping.items():
            msg = node.get("message")
            if not msg:
                continue

            role = msg.get("author", {}).get("role", "unknown")
            content = msg.get("content", {})

            # Extract text content
            text = ""
            if content.get("parts") and isinstance(content["parts"], list):
                text = " ".join(str(part) for part in content["parts"] if part).strip()
            elif content.get("text"):
                text = str(content["text"]).strip()
            elif isinstance(content, str):
                text = content.strip()

            # Filter and process text
            if text and len(text) >= MIN_TEXT_LENGTH:
                # Clean and truncate text
                text = clean_text(text)
                if len(text) > MAX_TEXT_LENGTH:
                    text = text[:MAX_TEXT_LENGTH] + "..."
                
                messages.append({
                    "id": msg.get("id", f"{conv_id}_{node_id}"),
                    "conversation_id": conv_id,
                    "conversation_title": title,
                    "role": role,
                    "text": text,
                    "account": account_label,
                    "create_time": create_time,
                    "node_id": node_id
                })
    
    print(f"Extracted {len(messages)} valid messages from {filepath}")
    return messages

def clean_text(text):
    """Clean text for better similarity matching."""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    return text.strip()

def advanced_categorize_message(text, title):
    """Advanced categorization with weighted keywords."""
    text_lower = (text + " " + title).lower()
    
    # Enhanced keyword categories with weights
    categories = {
        "technical": {
            "high": ["programming", "code", "python", "javascript", "api", "database", "algorithm", "debug", "error", "function", "software", "development"],
            "medium": ["tech", "computer", "system", "data", "web", "app", "coding", "script", "developer", "technical"],
            "low": ["digital", "online", "internet", "technology"]
        },
        "creative": {
            "high": ["story", "creative", "writing", "poem", "character", "plot", "design", "art", "music", "novel"],
            "medium": ["fiction", "narrative", "artistic", "imagination", "creativity", "drawing", "painting"],
            "low": ["idea", "inspiration", "beautiful", "aesthetic"]
        },
        "philosophy": {
            "high": ["philosophy", "ethics", "meaning", "existence", "consciousness", "moral", "belief", "truth", "reality"],
            "medium": ["philosophical", "metaphysics", "epistemology", "ontology", "wisdom", "virtue"],
            "low": ["think", "wonder", "question", "why"]
        },
        "personal": {
            "high": ["personal", "advice", "relationship", "career", "life", "decision", "feeling", "experience", "help me"],
            "medium": ["emotion", "family", "friend", "love", "stress", "worry", "goal", "dream"],
            "low": ["myself", "feel", "think", "want"]
        },
        "learning": {
            "high": ["learn", "explain", "understand", "how to", "what is", "tutorial", "guide", "course", "study"],
            "medium": ["education", "knowledge", "skill", "training", "practice", "lesson"],
            "low": ["know", "information", "about"]
        },
        "business": {
            "high": ["business", "marketing", "strategy", "company", "startup", "investment", "finance", "sales"],
            "medium": ["market", "customer", "revenue", "profit", "management", "entrepreneur"],
            "low": ["work", "job", "money", "economy"]
        },
        "health": {
            "high": ["health", "fitness", "exercise", "diet", "medical", "wellness", "mental health", "therapy"],
            "medium": ["doctor", "medicine", "nutrition", "workout", "physical", "psychological"],
            "low": ["body", "mind", "tired", "energy"]
        }
    }
    
    # Calculate weighted scores
    scores = {}
    for category, weight_groups in categories.items():
        score = 0
        for weight, keywords in weight_groups.items():
            weight_value = {"high": 3, "medium": 2, "low": 1}[weight]
            for keyword in keywords:
                if keyword in text_lower:
                    score += weight_value
        if score > 0:
            scores[category] = score
    
    # Return category with highest score
    if scores:
        return max(scores, key=scores.get)
    return "general"

def get_tfidf_embeddings(texts):
    """Generate TF-IDF embeddings for similarity calculation."""
    print(f"Generating TF-IDF embeddings for {len(texts)} messages...")
    
    # Use TF-IDF with custom parameters for better semantic understanding
    vectorizer = TfidfVectorizer(
        max_features=5000,  # Limit vocabulary size
        stop_words='english',
        ngram_range=(1, 2),  # Include both unigrams and bigrams
        min_df=2,  # Ignore terms that appear in less than 2 documents
        max_df=0.8,  # Ignore terms that appear in more than 80% of documents
        lowercase=True,
        strip_accents='unicode'
    )
    
    try:
        tfidf_matrix = vectorizer.fit_transform(texts)
        print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
        return tfidf_matrix.toarray()
    except Exception as e:
        print(f"Error generating TF-IDF embeddings: {e}")
        return np.zeros((len(texts), 100))  # Fallback

def create_keyword_based_connections(messages):
    """Create connections based on keyword overlap as fallback."""
    print("Creating keyword-based connections...")
    
    # Extract keywords from each message
    message_keywords = []
    for msg in messages:
        # Simple keyword extraction
        text = msg["text"].lower()
        # Remove common words and extract meaningful terms
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text)  # Words with 4+ characters
        word_freq = Counter(words)
        # Get top keywords
        keywords = [word for word, freq in word_freq.most_common(10)]
        message_keywords.append(set(keywords))
    
    edges = []
    for i in range(len(messages)):
        for j in range(i+1, len(messages)):
            # Calculate keyword overlap
            overlap = len(message_keywords[i] & message_keywords[j])
            total_keywords = len(message_keywords[i] | message_keywords[j])
            
            if total_keywords > 0:
                similarity = overlap / total_keywords
                if similarity > 0.2:  # Threshold for keyword similarity
                    edges.append({
                        "source": messages[i]["id"],
                        "target": messages[j]["id"],
                        "similarity": similarity,
                        "weight": similarity,
                        "type": "keyword"
                    })
    
    return edges

def create_similarity_graph(messages, embeddings=None):
    """Create nodes and edges based on similarity."""
    print("Building similarity graph...")
    
    nodes = []
    edges = []
    
    # Create nodes with enhanced categorization
    for i, msg in enumerate(messages):
        category = advanced_categorize_message(msg["text"], msg["conversation_title"])
        
        nodes.append({
            "id": msg["id"],
            "label": msg["text"][:100] + ("..." if len(msg["text"]) > 100 else ""),
            "title": msg["conversation_title"],
            "role": msg["role"],
            "account": msg["account"],
            "category": category,
            "full_text": msg["text"][:500],
            "create_time": msg["create_time"],
            "word_count": len(msg["text"].split())
        })
    
    # Create edges based on embeddings or keywords
    if embeddings is not None and embeddings.size > 0:
        print("Using TF-IDF similarity...")
        similarity_matrix = cosine_similarity(embeddings)
        
        edge_count = 0
        for i in range(len(messages)):
            for j in range(i+1, len(messages)):
                similarity = similarity_matrix[i, j]
                if similarity > SIM_THRESHOLD:
                    edges.append({
                        "source": messages[i]["id"],
                        "target": messages[j]["id"],
                        "similarity": float(similarity),
                        "weight": float(similarity),
                        "type": "tfidf"
                    })
                    edge_count += 1
    else:
        print("Falling back to keyword-based connections...")
        edges = create_keyword_based_connections(messages)
    
    print(f"Created {len(nodes)} nodes and {len(edges)} edges")
    return nodes, edges

def add_category_connections(nodes, edges):
    """Add connections between messages in the same category."""
    print("Adding category-based connections...")
    
    category_groups = {}
    for node in nodes:
        category = node["category"]
        if category not in category_groups:
            category_groups[category] = []
        category_groups[category].append(node["id"])
    
    category_edges = []
    for category, node_ids in category_groups.items():
        if len(node_ids) > 1:
            # Connect nodes within the same category (limited to avoid too many edges)
            for i, source in enumerate(node_ids[:20]):  # Limit to first 20 nodes per category
                for target in node_ids[i+1:min(i+6, len(node_ids))]:  # Connect to next 5 nodes
                    category_edges.append({
                        "source": source,
                        "target": target,
                        "similarity": 0.5,
                        "weight": 0.3,
                        "type": "category"
                    })
    
    print(f"Added {len(category_edges)} category-based edges")
    return edges + category_edges

def save_graph(nodes, edges, output_file):
    """Save the graph data to JSON."""
    # Calculate statistics
    category_stats = {}
    account_stats = {}
    
    for node in nodes:
        cat = node["category"]
        acc = node["account"]
        category_stats[cat] = category_stats.get(cat, 0) + 1
        account_stats[acc] = account_stats.get(acc, 0) + 1
    
    graph_data = {
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "similarity_threshold": SIM_THRESHOLD,
            "embedding_method": "TF-IDF + Keywords",
            "category_distribution": category_stats,
            "account_distribution": account_stats
        },
        "nodes": nodes,
        "links": edges
    }
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(graph_data, f, indent=2, ensure_ascii=False)
    
    print(f"Graph saved to {output_file}")

def main():
    """Main execution function."""
    print("ChatGPT Conversation Merger & Graph Generator (Offline Mode)")
    print("=" * 60)
    
    # Load messages from both accounts
    msgs_A = load_messages("conversations_A.json", "Account_A")
    msgs_B = load_messages("conversations_B.json", "Account_B")
    
    # Merge all messages
    all_msgs = msgs_A + msgs_B
    
    if not all_msgs:
        print("No messages found. Please check your export files.")
        return
    
    print(f"\nTotal messages loaded: {len(all_msgs)}")
    print(f"Account A: {len(msgs_A)} messages")
    print(f"Account B: {len(msgs_B)} messages")
    
    # Show category distribution
    categories = {}
    for msg in all_msgs:
        cat = advanced_categorize_message(msg["text"], msg["conversation_title"])
        categories[cat] = categories.get(cat, 0) + 1
    
    print(f"\nCategory distribution:")
    for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
        print(f"  {cat}: {count}")
    
    # Get TF-IDF embeddings
    texts = [msg["text"] for msg in all_msgs]
    embeddings = get_tfidf_embeddings(texts)
    
    # Create similarity graph
    nodes, edges = create_similarity_graph(all_msgs, embeddings)
    
    # Add category-based connections for better structure
    edges = add_category_connections(nodes, edges)
    
    # Save results
    save_graph(nodes, edges, OUTPUT_FILE)
    
    print(f"\n✅ Successfully created merged conversation graph!")
    print(f"📊 Statistics:")
    print(f"   - Total conversations processed: {len(set(msg['conversation_id'] for msg in all_msgs))}")
    print(f"   - Total messages: {len(all_msgs)}")
    print(f"   - Graph nodes: {len(nodes)}")
    print(f"   - Graph edges: {len(edges)}")
    print(f"   - Average edges per node: {len(edges) / len(nodes) if nodes else 0:.2f}")
    print(f"   - Output file: {OUTPUT_FILE}")
    
    # Show top categories
    category_counts = {}
    for node in nodes:
        cat = node["category"]
        category_counts[cat] = category_counts.get(cat, 0) + 1
    
    print(f"\n📋 Top Categories:")
    for cat, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"   {cat}: {count} messages")

if __name__ == "__main__":
    main()