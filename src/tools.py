"""
Tools Module - Explicit callable tools the agent can choose from.
"""

import time
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


@dataclass
class Tool:
    """Represents a callable tool with metadata."""
    name: str
    description: str
    function: Callable
    required_inputs: List[str]
    output_type: str


class ToolRegistry:
    """Registry of available tools for the agent."""
    
    def __init__(self):
        self._tools: Dict[str, Tool] = {}
        self._register_default_tools()
    
    def _register_default_tools(self):
        self.register(Tool(
            name="analyze_data_structure",
            description="Analyze the structure and characteristics of input data",
            function=analyze_data_structure,
            required_inputs=["data_path"],
            output_type="dict"
        ))
        self.register(Tool(
            name="summarize_text",
            description="Summarize a collection of text responses into key themes",
            function=summarize_text,
            required_inputs=["texts"],
            output_type="str"
        ))
        self.register(Tool(
            name="cluster_feedback",
            description="Cluster qualitative feedback into thematic groups using ML",
            function=cluster_feedback,
            required_inputs=["texts"],
            output_type="dict"
        ))
        self.register(Tool(
            name="analyze_sentiment_distribution",
            description="Analyze the distribution of sentiment/scores across responses",
            function=analyze_sentiment_distribution,
            required_inputs=["data_path", "score_columns"],
            output_type="dict"
        ))
        self.register(Tool(
            name="generate_recommendations",
            description="Generate actionable recommendations based on insights",
            function=generate_recommendations,
            required_inputs=["insights"],
            output_type="list"
        ))
        self.register(Tool(
            name="extract_key_phrases",
            description="Extract key phrases and frequent terms from text",
            function=extract_key_phrases,
            required_inputs=["texts"],
            output_type="list"
        ))
    
    def register(self, tool: Tool):
        self._tools[tool.name] = tool
    
    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)
    
    def list_tools(self) -> List[str]:
        return list(self._tools.keys())
    
    def get_descriptions(self) -> Dict[str, str]:
        return {name: tool.description for name, tool in self._tools.items()}
    
    def execute(self, name: str, **kwargs) -> Dict[str, Any]:
        tool = self.get(name)
        if not tool:
            return {"success": False, "error": f"Tool '{name}' not found", "output": None}
        
        start_time = time.time()
        try:
            result = tool.function(**kwargs)
            execution_time = (time.time() - start_time) * 1000
            return {"success": True, "output": result, "execution_time_ms": execution_time, "error": None}
        except Exception as e:
            return {"success": False, "output": None, "error": str(e), "execution_time_ms": (time.time() - start_time) * 1000}


def analyze_data_structure(data_path: str) -> Dict[str, Any]:
    """Analyze the structure of a CSV file."""
    df = pd.read_csv(data_path)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    text_cols = df.select_dtypes(include=['object']).columns.tolist()
    likert_cols = [col for col in numeric_cols if df[col].min() >= 1 and df[col].max() <= 5]
    feedback_cols = [col for col in text_cols if df[col].str.len().mean() > 50]
    
    return {
        "row_count": len(df),
        "column_count": len(df.columns),
        "columns": df.columns.tolist(),
        "numeric_columns": numeric_cols,
        "text_columns": text_cols,
        "likely_likert_columns": likert_cols,
        "likely_feedback_columns": feedback_cols,
        "sample_row": df.iloc[0].to_dict() if len(df) > 0 else {},
        "missing_values": df.isnull().sum().to_dict()
    }


def summarize_text(texts: List[str], max_themes: int = 5) -> str:
    """Summarize texts into key themes."""
    if not texts:
        return "No texts provided for summarization."
    
    word_freq = {}
    for text in texts:
        for word in text.lower().split():
            if len(word) > 4:
                word_freq[word] = word_freq.get(word, 0) + 1
    
    top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
    summary = f"Analyzed {len(texts)} text responses.\n"
    summary += f"Most frequent terms: {', '.join([w[0] for w in top_words])}\n"
    summary += f"Average response length: {sum(len(t) for t in texts) / len(texts):.0f} characters"
    return summary


def cluster_feedback(texts: List[str], n_clusters: int = 4) -> Dict[str, Any]:
    """Cluster feedback into thematic groups using TF-IDF and KMeans."""
    if len(texts) < n_clusters:
        return {"error": "Not enough texts for clustering", "clusters": []}
    
    vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(texts)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(tfidf_matrix)
    
    cluster_results = {}
    for i in range(n_clusters):
        cluster_texts = [texts[j] for j in range(len(texts)) if clusters[j] == i]
        if cluster_texts:
            cluster_vec = vectorizer.transform(cluster_texts)
            mean_tfidf = cluster_vec.mean(axis=0).A1
            top_indices = mean_tfidf.argsort()[-5:][::-1]
            top_terms = [vectorizer.get_feature_names_out()[idx] for idx in top_indices]
        else:
            top_terms = []
        
        cluster_results[f"cluster_{i}"] = {
            "count": len(cluster_texts),
            "top_terms": top_terms,
            "sample_texts": cluster_texts[:3]
        }
    
    return {"n_clusters": n_clusters, "total_texts": len(texts), "clusters": cluster_results}


def analyze_sentiment_distribution(data_path: str, score_columns: List[str]) -> Dict[str, Any]:
    """Analyze distribution of scores across specified columns."""
    df = pd.read_csv(data_path)
    results = {}
    
    for col in score_columns:
        if col in df.columns:
            results[col] = {
                "mean": float(df[col].mean()),
                "median": float(df[col].median()),
                "std": float(df[col].std()),
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "distribution": df[col].value_counts().sort_index().to_dict()
            }
    
    if results:
        overall_mean = np.mean([r["mean"] for r in results.values()])
        results["overall"] = {
            "average_across_metrics": overall_mean,
            "interpretation": "positive" if overall_mean > 3.5 else "neutral" if overall_mean > 2.5 else "negative"
        }
    
    return results


def generate_recommendations(insights: List[str], max_recommendations: int = 5) -> List[Dict[str, str]]:
    """Generate recommendations based on insights."""
    if not insights:
        return [{"recommendation": "Gather more data before making recommendations", "priority": "high"}]
    
    recommendations = []
    for i, insight in enumerate(insights[:max_recommendations]):
        recommendations.append({
            "recommendation": f"Address issue identified: {insight[:100]}...",
            "priority": "high" if i < 2 else "medium",
            "expected_impact": "Improvement in customer satisfaction",
            "effort": "medium"
        })
    return recommendations


def extract_key_phrases(texts: List[str], top_n: int = 20) -> List[Dict[str, Any]]:
    """Extract key phrases from texts using TF-IDF."""
    if not texts:
        return []
    
    vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=100, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(texts)
    
    mean_scores = tfidf_matrix.mean(axis=0).A1
    feature_names = vectorizer.get_feature_names_out()
    sorted_indices = mean_scores.argsort()[::-1][:top_n]
    
    return [{"phrase": feature_names[i], "score": float(mean_scores[i])} for i in sorted_indices]


tool_registry = ToolRegistry()
